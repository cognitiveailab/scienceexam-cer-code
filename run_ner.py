from __future__ import absolute_import, division, print_function
import math
import argparse
import glob
import json
import logging
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_transformers import (WEIGHTS_NAME, AdamW, BertConfig,
                                  BertForTokenClassification, BertTokenizer,
                                  WarmupLinearSchedule)
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from printOutput import token_predition_write
from multi_label_ner_performance import classification_report
from Data_process.plain2conll import process_plain_text

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class Ner(BertForTokenClassification):

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,valid_ids=None,attention_mask_label=None):
        sequence_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32,device='cuda')
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)
        if labels is not None:
            labels = labels.float()
            loss_fct = nn.BCEWithLogitsLoss()
            # Only keep active parts of the loss
            attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:
            return logits


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask

def readfile(filename):
    '''
    read file
    '''
    f = open(filename)
    data = []
    sentence = []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append([i.strip() for i in splits[5:]])
    if len(sentence) >0:
        data.append((sentence,label))
        sentence = []
        label = []
    return data

def readfile_arc(filename):
    '''
    read file
    '''
    f = open(filename)
    data = []
    sentence = []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(['O'])
    if len(sentence) >0:
        data.append((sentence,label))
        sentence = []
        label = []
    return data

def readfile_plain_text(data_dir, plain_text):
    '''
    read file
    '''
    process_plain_text(data_dir, plain_text)
    base_filename = os.path.basename(plain_text)
    f = open(os.path.join(data_dir, 'conll_'+base_filename))
    data = []
    sentence = []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(['O'])
    if len(sentence) >0:
        data.append((sentence,label))
        sentence = []
        label = []
    return data

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
        
    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_ARC_test_examples(self, data_dir):
        return NotImplementedError()

    def get_plain_text_examples(self, data_dir, plain_text_data):
        return NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)

    @classmethod
    def _read_arc_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile_arc(input_file)

    @classmethod
    def _read_plain_text(cls, data_dir, plain_text, quotechar=None):
        """Reads a tab separated value file."""
        return readfile_plain_text(data_dir, plain_text)


class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_spacy.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid_spacy.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_spacy.txt")), "test")

    def get_ARC_test_examples(self, data_dir):
        return self._create_arc_examples(self._read_arc_tsv(os.path.join(data_dir, "ARC_test_spacy.txt")), "test")

    def get_plain_text_examples(self, data_dir, plain_text_data):
        return self._create_plain_text_examples(self._read_plain_text(data_dir, plain_text_data), "test")


    def get_labels(self):
        return ['B-TypesOfEvent', 'I-OtherAnimalProperties', 'I-NUMBER', 'B-FossilTypesIndexFossil', 'I-Fungi', 'I-Verify', 'I-AbsorbEnergy', 'I-Conductivity', 'I-States', 'B-LivingThing', 'B-Patents', 'B-Circuits', 'B-MeasurementsForHeatChange', 'B-RepresentingElementsAndMolecules', 'B-IllnessPreventionCuring', 'B-Release', 'B-Directions', 'B-Comet', 'B-PartsOfChemicalReactions', 'B-Asteroid', 'B-TimeMeasuringTools', 'B-TidesHighTideLowTide', 'I-MetalSolids', 'B-Represent', 'I-PlantCellPart', 'I-NaturalSelection', 'I-EcosystemsEnvironment', 'B-SystemProcessStages', 'B-Energy', 'B-Occupation', 'I-SeasonsFallAutumnWinterSpringSummer', 'B-Evolution', 'I-TraceFossil', 'B-ScientificMeetings', 'I-PartsOfTheExcretorySystem', 'B-TectonicPlates', 'B-ExamplesOfSounds', 'B-Comparisons', 'I-AvoidReject', 'B-ManmadeObjects', 'B-ClassesOfElements', 'I-CelestialMeasurements', 'I-VerbsForLocate', 'B-Inertia', 'I-Bacteria', 'I-Mutation', 'B-Classify', 'B-ProduceEnergy', 'B-Metabolism', 'I-Numbers', 'B-NaturalPhenomena', 'I-BusinessNames', 'I-Negations', 'I-PopularMedia', 'B-Complexity', 'I-Sickness', 'B-PlantCellPart', 'I-NutritiveSubstancesForAnimalsOrPlants', 'B-TIME', 'I-ColorChangingActions', 'B-PushingActions', 'I-ProduceEnergy', 'I-WeatherPhenomena', 'B-CelestialEvents', 'B-GalaxyParts', 'B-SystemOfCommunication', 'B-Goal', 'B-ExcretoryActions', 'I-Substances', 'I-GroupsOfOrganisms', 'B-Collect', 'B-SpaceAgencies', 'B-SyntheticMaterial', 'I-FossilFuel', 'B-AmphibianAnimalPart', 'B-PhysicalProperty', 'I-AnimalAdditionalCategories', 'I-Source', 'I-GeometricMeasurements', 'I-Relevant', 'I-Permit', 'I-GeographicFormationParts', 'I-ElectricityAndCircuits', 'B-GenericTerms', 'B-PhaseTransitionPoint', 'B-ArcheologicalProcessTechnique', 'B-Property', 'I-Result', 'B-Health', 'I-Mass', 'B-FormChangingActions', 'B-CarbonCycle', 'B-Protist', 'I-Release', 'B-Vehicle', 'I-EndocrineSystem', 'I-RelativeNumber', 'I-CapillaryAction', 'B-MammalAnimalPart', 'I-ImmuneSystem', 'I-ActionsForAnimals', 'B-Examine', 'B-LunarPhases', 'I-PartsOfABuilding', 'B-Experimentation', 'B-PartsOfTheNervousSystem', 'I-CirculatorySystem', 'I-ORGANIZATION', 'B-Soil', 'B-Angiosperm', 'I-PullingActions', 'B-TrueFormFossil', 'B-ActionsForNutrition', 'B-ExamplesOfHabitats', 'B-ClothesTextiles', 'I-NaturalResources', 'B-ReproductiveSystem', 'B-PERCENT', 'B-WordsForData', 'I-Extinction', 'B-DigestiveSubstances', 'I-Agriculture', 'B-ActionsForAnimals', 'I-Amphibian', 'I-Property', 'B-Sky', 'I-Minerals', 'I-NaturalPhenomena', 'B-GeographicFormationParts', 'B-Start', 'I-CastFossilMoldFossil', 'B-Meals', 'B-MoneyTerms', 'I-PartsOfEarthLayers', 'B-Depth', 'B-MeasuringSpeed', 'B-PH', 'B-PlantNutrients', 'B-AtomComponents', 'I-OuterPlanets', 'B-Choose', 'B-Relevant', 'B-ChemicalProduct', 'I-DistanceComparison', 'I-VariablesControls', 'B-SeasonsFallAutumnWinterSpringSummer', 'I-RespiratorySystem', 'B-PrenatalOrganismStates', 'B-Viewpoint', 'I-PartsOfABusiness', 'I-ConstructiveDestructiveForces', 'B-EarthPartsGrossGroundAtmosphere', 'I-DigestionActions', 'B-Shape', 'I-GeneticProperty', 'B-Reactions', 'I-ImportanceComparison', 'I-TimeUnit', 'I-TechnologicalComponent', 'B-MeteorologicalModels', 'B-Medicine', 'I-Injuries', 'I-ScientificMethod', 'I-StructuralAdaptation', 'B-Meteor', 'B-PullingForces', 'I-Continents', 'I-SpaceAgencies', 'B-OrganicProcesses', 'B-Growth', 'I-Habitat', 'I-Sedimentary', 'I-Width', 'B-SpeedUnit', 'B-HumanPart', 'I-Medicine', 'I-Separation', 'B-CoolingToolsFood', 'I-Galaxy', 'I-Require', 'B-Development', 'B-VisualComparison', 'I-Language', 'B-Groups', 'B-Method', 'B-MetalSolids', 'I-AtmosphericLayers', 'B-TemporalProperty', 'B-AnimalPart', 'B-Mixtures', 'I-WaterVehiclePart', 'B-Pattern', 'B-PartsOfTheCirculatorySystem', 'I-AmountChangingActions', 'B-ElectricalEnergySource', 'B-OtherAnimalProperties', 'B-CelestialObject', 'I-Gene', 'B-ORGANIZATION', 'I-TemporalProperty', 'I-Transportation', 'O', 'I-Asteroid', 'I-PH', 'B-BirdAnimalPart', 'B-Value', 'B-OrganicCompounds', 'B-FiltrationTool', 'I-Event', 'B-BusinessNames', 'I-LearnedBehavior', 'I-Vehicle', 'I-BeliefKnowledge', 'B-PressureMeasuringTool', 'I-OtherHumanProperties', 'B-Agriculture', 'I-PartsOfTheCirculatorySystem', 'B-Continents', 'B-ThermalEnergy', 'B-EnvironmentalPhenomena', 'B-GroupsOfOrganisms', 'B-Spectra', 'B-SpaceMissionsEGApolloGeminiMercury', 'B-Insect', 'B-Buy', 'B-ImmuneSystem', 'B-AtmosphericLayers', 'I-OrganismRelationships', 'I-ReleaseEnergy', 'I-Occupation', 'I-Meteorology', 'B-RelativeNumber', 'B-Differentiate', 'B-Planet', 'B-PartsOfTheFoodChain', 'B-MolecularProperties', 'B-NUMBER', 'B-PopularMedia', 'I-EnergyWaves', 'I-PERSON', 'B-NaturalResources', 'I-Particles', 'B-Day', 'B-VisualProperty', 'B-Genetics', 'I-MassMeasuringTool', 'B-FoodChain', 'I-ClothesTextiles', 'B-Group', 'B-EnvironmentalDamageDestruction', 'I-ElementalComponents', 'B-MarkersOfTime', 'B-TypesOfChemicalReactions', 'B-PartsOfTheIntegumentarySystem', 'B-ImportanceComparison', 'B-PartsOfABusiness', 'B-ObjectPart', 'I-CellsAndGenetics', 'B-Employment', 'B-PlantProcesses', 'I-MagneticDevice', 'I-PowerUnit', 'B-ObjectQuantification', 'I-LiquidHoldingContainersRecepticles', 'B-Composition', 'B-TraceFossil', 'I-Viewpoint', 'I-SystemAndFunctions', 'I-ConstructionTools', 'B-ElectromagneticSpectrum', 'I-ManmadeObjects', 'I-Spectra', 'I-Cycles', 'B-ElementalComponents', 'B-Result', 'I-VisualProperty', 'B-Human', 'I-Plant', 'I-ComputingDevice', 'B-NonlivingPartsOfTheEnvironment', 'B-GeneticProperty', 'B-PropertiesOfFood', 'B-FossilFuel', 'I-ArcheologicalProcessTechnique', 'B-Fossils', 'B-AmountChangingActions', 'I-Protist', 'I-OtherGeographicWords', 'B-Arachnid', 'B-CelestialMovement', 'B-BlackHole', 'I-ElectricityGeneration', 'I-SeparatingMixtures', 'I-VolumeUnit', 'I-FossilForming', 'B-SkeletalSystem', 'B-LiquidMovement', 'I-EclipseEvents', 'I-Aquatic', 'B-ParticleMovement', 'I-Star', 'I-PropertyOfProduction', 'I-PropertyOfMotion', 'B-PartsOfDNA', 'B-Separate', 'B-Substances', 'I-TectonicPlates', 'I-Inheritance', 'I-Energy', 'B-PhaseChangingActions', 'B-BusinessIndustry', 'B-Metamorphic', 'I-PhasesOfWater', 'B-ChangeInLocation', 'I-PartsOfTheRespiratorySystem', 'I-ChangeInto', 'B-MeasuresOfAmountOfLight', 'B-AnimalCellPart', 'B-Compound', 'B-PERSON', 'I-ElectricalEnergySource', 'B-ReleaseEnergy', 'I-GeologicTheories', 'B-WrittenMedia', 'I-Position', 'I-ScientificMeetings', 'I-Angiosperm', 'I-EnvironmentalPhenomena', 'B-OtherDescriptionsForPlantsBiennialLeafyEtc', 'B-AbsorbEnergy', 'B-MagneticDevice', 'B-YearNumerals', 'B-Observe', 'I-PartsOfTheSkeletalSystem', 'I-Planet', 'B-AmountComparison', 'I-Quality', 'B-PartsOfARepresentation', 'B-ChemicalProperty', 'B-Language', 'I-PartsOfTheMuscularSystem', 'B-Cell', 'I-Senses', 'B-Material', 'B-Rarity', 'I-Help', 'B-IntegumentarySystem', 'I-TypeOfConsumer', 'B-Cities', 'I-PhysicalActivity', 'B-AvoidReject', 'I-Forests', 'B-WeatherPhenomena', 'B-Device', 'I-TechnologicalInstrument', 'I-ManMadeGeographicFormations', 'B-DwarfPlanets', 'I-Use', 'B-Vacuum', 'B-ElectricalProperty', 'B-AcademicMedia', 'B-Birth', 'B-OuterPlanets', 'I-PullingForces', 'B-GuidelinesAndRules', 'I-CelestialLightOnEarth', 'B-TemperatureMeasuringTools', 'I-Validity', 'I-Rock', 'I-ObservationInstrumentsTelescopeBinoculars', 'I-LightMovement', 'I-Comparisons', 'B-PlantPart', 'B-PerformingResearch', 'I-EnvironmentalDamageDestruction', 'I-HeatingAppliance', 'I-OrganicCompounds', 'I-ObservationPlacesEGObservatory', 'B-GeologicTheories', 'B-Difficulty', 'B-Communicate', 'B-Extinction', 'I-CleanUp', 'I-Observe', 'B-PartsOfAGroup', 'I-SpaceProbes', 'B-WeightMeasuringTool', 'I-UnderwaterEcosystem', 'B-ChangeInComposition', 'I-ChemicalProcesses', 'I-PartsOfTheNervousSystem', 'B-CardinalDirectionsNorthEastSouthWest', 'I-NaturalMaterial', 'B-Measurements', 'I-FoodChain', 'I-VisualComparison', 'B-Sedimentary', 'B-Quality', 'B-StateOfMatter', 'B-TransferEnergy', 'B-SoundProducingObject', 'I-CarbonCycle', 'I-FiltrationTool', 'B-Uptake', 'B-Wetness', 'B-FossilLocationImplicationsOfLocationEXMarineAnimalFossilsInTheGrandCanyon', 'I-DigestiveSubstances', 'I-MagneticForce', 'I-PushingForces', 'B-Response', 'I-SoundProducingObject', 'I-LivingThing', 'B-Succeed', 'I-Move', 'I-Shape', 'I-Rarity', 'B-MuscularSystemActions', 'I-Identify', 'I-VolumeMeasuringTool', 'B-ElectricalUnit', 'I-Nebula', 'B-Safety', 'B-Animal', 'B-Verify', 'B-GeographicFormations', 'B-Audiences', 'I-Harm', 'B-Foods', 'I-IncreaseDecrease', 'I-ScientificTools', 'B-Occur', 'B-AstronomicalDistanceUnitsLightYearAstronomicalUnitAu', 'I-MarkersOfTime', 'I-Locations', 'I-SkeletalSystem', 'I-Color', 'B-OtherEnergyResources', 'B-Meteorology', 'B-Flammability', 'B-ElectricalEnergy', 'B-PartsOfTheDigestiveSystem', 'I-Composition', 'B-Classification', 'I-Insect', 'I-GeneticProcesses', 'I-Height', 'I-SimpleMachines', 'I-PropertiesOfSickness', 'I-AcademicMedia', 'B-Identify', 'B-VehicularSystemsParts', 'B-Indicate', 'I-MineralProperties', 'I-Size', 'I-MuscularSystem', 'B-Use', 'B-PropertiesOfSickness', 'B-AbilityAvailability', 'I-MineralFormations', 'B-Separation', 'B-Calculations', 'B-PropertiesOfWaves', 'B-DATE', 'I-TIME', 'I-TypesOfIllness', 'I-Fossils', 'B-Undiscovered', 'I-Taxonomy', 'B-PartsOfObservationInstruments', 'B-Nutrition', 'B-Cycles', 'B-ResistanceStrength', 'I-PhaseChanges', 'I-GeographicFormations', 'B-AnimalSystemsProcesses', 'B-CapillaryAction', 'B-PropertyOfMotion', 'B-FossilRecordTimeline', 'B-Taxonomy', 'B-Require', 'I-PlantNutrients', 'B-Break', 'I-PartsOfBodiesOfWater', 'B-Gymnosperm', 'B-SpecificNamedBodiesOfWater', 'B-WaitStay', 'I-ChemicalProperty', 'I-Biology', 'B-GeopoliticalLocations', 'I-RelativeTime', 'B-PartsOfTheReproductiveSystem', 'I-Distance', 'B-AnimalAdditionalCategories', 'B-ManMadeGeographicFormations', 'B-LOCATION', 'I-WrittenMedia', 'I-WaterVehicle', 'B-Unknown', 'I-Sky', 'B-Frequency', 'B-ComputingDevice', 'I-Cell', 'B-Homeostasis', 'I-AtomicProperties', 'B-PullingActions', 'B-BacteriaPart', 'B-EclipseEvents', 'B-Brightness', 'I-Classify', 'B-Plant', 'B-WaterVehiclePart', 'I-Age', 'I-ActionsForNutrition', 'I-ObjectPart', 'B-Touch', 'I-Alter', 'I-Response', 'I-EndocrineActions', 'B-Size', 'I-Goal', 'B-LightProducingObject', 'B-Rock', 'I-PartsOfDNA', 'B-PerformAnActivity', 'B-DistanceUnit', 'I-Reproduction', 'I-MedicalTerms', 'B-WeatherDescriptions', 'B-PoorHealth', 'B-SolidMatter', 'B-Stability', 'I-LiquidMatter', 'I-Igneous', 'B-OrganismRelationships', 'I-Frequency', 'B-DURATION', 'I-GeographicFormationProcess', 'B-AquaticAnimalPart', 'I-MuscularSystemActions', 'B-ManmadeLocations', 'B-ColorChangingActions', 'B-Appliance', 'I-Animal', 'B-Permit', 'B-ResultsOfDecomposition', 'I-Traffic', 'B-PartsOfTheMuscularSystem', 'I-VehicularSystemsParts', 'B-Advertising', 'B-PartsOfAChromosome', 'B-Cause', 'I-Mammal', 'B-ChemicalProcesses', 'B-TypesOfWaterInBodiesOfWater', 'I-Represent', 'I-Device', 'I-Homeostasis', 'B-Minerals', 'B-Hardness', 'B-Texture', 'B-BodiesOfWater', 'B-ReplicatingResearch', 'B-LevelOfInclusion', 'I-OtherEnergyResources', 'I-EnergyUnit', 'B-AtomicProperties', 'B-Matter', 'B-SpacecraftHumanRated', 'B-SimpleMachines', 'I-ClassesOfElements', 'B-Currents', 'B-TheUniverseUniverseAndItsParts', 'B-Representation', 'B-EndocrineActions', 'I-ExamplesOfSounds', 'B-CleanUp', 'B-SeparatingMixtures', 'I-TransferEnergy', 'B-RelativeDirection', 'B-Precipitation', 'I-ConservationLaws', 'B-Particles', 'B-PrepositionalDirections', 'I-Communicate', 'B-Transportation', 'B-TemperatureUnit', 'I-LayersOfTheEarth', 'I-Gymnosperm', 'I-Separate', 'I-BlackHole', 'I-TheUniverseUniverseAndItsParts', 'I-Stability', 'B-Gender', 'I-ThermalEnergy', 'B-Help', 'I-PartsOfTheImmuneSystem', 'I-Create', 'B-BeliefKnowledge', 'B-Eukaryote', 'I-PartsOfTheDigestiveSystem', 'I-SoundEnergy', 'B-Age', 'B-Believe', 'I-Collect', 'B-TheoryOfPhysics', 'B-QualityComparison', 'B-SafetyEquipment', 'B-SystemParts', 'B-FossilForming', 'I-DigestiveSystem', 'I-Calculations', 'I-Relations', 'B-Consumption', 'I-FormChangingActions', 'B-EnergyWaves', 'I-CosmologicalTheoriesBigBangBigCrunch', 'I-ChemicalProduct', 'B-RelativeTime', 'B-InsectAnimalPart', 'B-OutbreakClassification', 'B-ScientificTheoryExperimentationAndHistory', 'B-PerformingExperimentsWell', 'I-OrganicProcesses', 'B-PercentUnit', 'I-Speed', 'B-ContainBeComposedOf', 'B-BehavioralAdaptation', 'B-LearnedBehavior', 'I-StarLayers', 'I-SpecificNamedBodiesOfWater', 'B-MineralFormations', 'I-AnimalClassificationMethod', 'B-ElectricityAndCircuits', 'B-AstronomyAeronautics', 'B-MagneticDirectionMeasuringTool', 'B-CirculatorySystem', 'B-CookingToolsFood', 'B-GeneticRelations', 'I-PartsOfTheReproductiveSystem', 'B-PartsOfWaterCycle', 'B-SystemAndFunctions', 'B-MagneticEnergy', 'I-Divide', 'I-ExcretoryActions', 'I-Cause', 'B-ActionsForTides', 'I-ViewingTools', 'B-StructuralAdaptation', 'B-MassMeasuringTool', 'I-PlantPart', 'I-AnimalPart', 'B-LiquidHoldingContainersRecepticles', 'B-LandVehicle', 'B-Preserve', 'I-ConcludingResearch', 'I-Wetness', 'I-AirVehicle', 'B-Injuries', 'B-InnerPlanets', 'B-DistanceComparison', 'I-SensoryTerms', 'B-ActionsForAgriculture', 'B-PartsOfEarthLayers', 'I-RespirationActions', 'I-AnimalSystemsProcesses', 'B-PartsOfWaves', 'B-TimeUnit', 'I-Appliance', 'I-SpaceMissionsEGApolloGeminiMercury', 'B-InheritedBehavior', 'I-TemperatureMeasuringTools', 'B-AnalyzingResearch', 'B-EndocrineSystem', 'I-StopRemove', 'B-TypesOfTerrestrialEcosystems', 'B-Source', 'B-ActUponSomething', 'I-GalaxyParts', 'B-ElectricityMeasuringTool', 'I-Bird', 'I-MechanicalMovement', 'I-Pressure', 'B-NationalityOrigin', 'B-Mutation', 'B-SpacecraftSubsystem', 'B-CelestialLightOnEarth', 'B-GeometricUnit', 'I-WaitStay', 'I-ActionsForTides', 'B-Problem', 'B-Blood', 'B-AnimalClassificationMethod', 'I-Soil', 'I-LandVehicle', 'B-ApparentCelestialMovement', 'B-Compete', 'I-ProbabilityAndCertainty', 'I-Choose', 'B-GeographicFormationProcess', 'I-Reactions', 'I-SystemOfCommunication', 'I-FeedbackMechanism', 'B-RelativeLocations', 'I-Temperature', 'I-TidesHighTideLowTide', 'I-WeightMeasuringTool', 'B-MuscularSystem', 'B-Rigidity', 'B-Satellite', 'I-Currents', 'I-ChemicalChange', 'I-Start', 'B-PartsOfRNA', 'B-VolumeMeasuringTool', 'I-ElectricalEnergy', 'B-PhasesOfWater', 'I-RepresentingElementsAndMolecules', 'B-MassUnit', 'I-Material', 'I-LunarPhases', 'B-TheoryOfMatter', 'B-Width', 'I-Experimentation', 'I-Measurements', 'B-States', 'I-IntegumentarySystem', 'B-Mammal', 'I-MeteorologicalModels', 'I-Associate', 'B-CombineAdd', 'I-Products', 'I-GeneticRelations', 'B-GranularSolids', 'B-ElectricityGeneration', 'I-HardnessUnit', 'B-CellsAndGenetics', 'B-Locations', 'I-Adaptation', 'B-Forests', 'I-FossilLocationImplicationsOfLocationEXMarineAnimalFossilsInTheGrandCanyon', 'B-Negations', 'I-LocationChangingActions', 'I-MassUnit', 'B-Conductivity', 'I-SolarSystem', 'I-CookingToolsFood', 'I-GaseousMatter', 'I-TypesOfTerrestrialEcosystems', 'I-TimeMeasuringTools', 'B-LightMovement', 'B-LivingDying', 'B-ConservationLaws', 'I-Nutrition', 'B-SubstancesProducedByPlantProcesses', 'B-GroupsOfScientists', 'I-WavePerception', 'B-ResponseType', 'I-NonlivingPartsOfTheEnvironment', 'B-PlanetParts', 'B-LiquidMatter', 'B-PropertyOfProduction', 'B-IncreaseDecrease', 'B-TerrestrialLocations', 'B-Relations', 'B-Height', 'I-BacteriaPart', 'I-Element', 'B-Element', 'I-PhysicalProperty', 'I-Precipitation', 'B-Monera', 'B-SeedlessVascular', 'I-Discovery', 'I-EarthPartsGrossGroundAtmosphere', 'B-VerbsForLocate', 'B-Toxins', 'B-Permeability', 'B-Star', 'I-PhaseChangingActions', 'I-Human', 'B-Year', 'B-WordsForOffspring', 'I-Compound', 'I-GroupsOfScientists', 'I-ElectricAppliance', 'B-Igneous', 'B-ChemicalChange', 'I-PartsOfAVirus', 'B-EmergencyServices', 'B-Move', 'B-HardnessUnit', 'B-EcosystemsEnvironment', 'B-CoolingAppliance', 'I-TrueFormFossil', 'B-Mass', 'I-NationalityOrigin', 'B-Nebula', 'B-StopRemove', 'I-Behaviors', 'B-Hypothesizing', 'B-Traffic', 'B-QuestionActivityType', 'B-Adaptation', 'I-ReproductiveSystem', 'I-InsectAnimalPart', 'I-FossilTypesIndexFossil', 'B-PostnatalOrganismStages', 'B-WaterVehicle', 'B-ReptileAnimalPart', 'I-SafetyEquipment', 'B-Sickness', 'B-PhysicalChange', 'I-Metabolism', 'I-LightProducingObject', 'I-Unknown', 'B-PartsOfEndocrineSystem', 'B-Biology', 'B-PressureUnit', 'I-ElectricalUnit', 'B-ConcludingResearch', 'B-TechnologicalComponent', 'B-GeometricSpatialObjects', 'I-OtherOrganismProperties', 'I-PERCENT', 'I-ChangeInLocation', 'B-ConstructionTools', 'I-AstronomicalDistanceUnitsLightYearAstronomicalUnitAu', 'I-Genetics', 'B-PhysicalActivity', 'B-ORDINAL', 'B-NaturalSelection', 'I-PhaseTransitionPoint', 'B-ForceUnit', 'B-LayersOfTheEarth', 'I-ScientificAssociationsAdministrations', 'I-Groups', 'I-NorthernHemisphereLocations', 'I-CirculationActions', 'B-Changes', 'I-TheoryOfPhysics', 'B-RespiratorySystem', 'B-LightExaminingTool', 'I-NervousSystem', 'I-ActUponSomething', 'I-Constellation', 'I-SouthernHemisphereLocations', 'B-ArithmeticMeasure', 'I-EmergencyServices', 'B-OtherOrganismProperties', 'B-PartsOfBodiesOfWater', 'I-Cost', 'B-Exemplar', 'I-SpaceVehicle', 'I-CellProcesses', 'B-SpaceVehicle', 'B-PartsOfTheExcretorySystem', 'B-DigestiveSystem', 'I-LivingDying', 'B-SoundEnergy', 'B-Inheritance', 'I-WordsForData', 'B-SouthernHemisphereLocations', 'I-Believe', 'B-Validity', 'I-HumanPart', 'B-GeneticProcesses', 'B-CosmologicalTheoriesBigBangBigCrunch', 'B-WordsRelatingToCosmologicalTheoriesExpandContract', 'I-AquaticAnimalPart', 'I-ExamplesOfHabitats', 'I-BodiesOfWater', 'I-Blood', 'B-ViewingTools', 'B-NaturalMaterial', 'B-GaseousMovement', 'B-Death', 'B-Surpass', 'B-PowerUnit', 'B-ObservationInstrumentsTelescopeBinoculars', 'I-ResultsOfDecomposition', 'I-ApparentCelestialMovement', 'B-ElectricAppliance', 'B-Geography', 'B-Bird', 'I-DensityUnit', 'B-GapsAndCracks', 'B-OtherProperties', 'B-Harm', 'I-Development', 'I-SolidMatter', 'B-PartsOfAVirus', 'I-ElectromagneticSpectrum', 'B-Actions', 'B-Discovery', 'B-OpportunitiesAndTheirExtent', 'I-Permeability', 'I-AreaUnit', 'I-ContainBeComposedOf', 'B-TypesOfIllness', 'I-Safety', 'B-SensoryTerms', 'I-DistanceMeasuringTools', 'B-AngleMeasuringTools', 'I-LifeCycle', 'B-PartsOfTheSkeletalSystem', 'I-WordsRelatingToCosmologicalTheoriesExpandContract', 'B-Position', 'I-GeologicalEonsErasPeriodsEpochsAges', 'B-FrequencyUnit', 'B-StateOfBeing', 'B-Temperature', 'B-DigestionActions', 'B-Habitat', 'B-Gene', 'I-LevelOfInclusion', 'I-Force', 'B-ObservationPlacesEGObservatory', 'I-CoolingAppliance', 'B-Ability', 'I-Countries', 'B-Alter', 'B-Reptile', 'B-LifeCycle', 'B-Cost', 'I-PerformAnActivity', 'B-SolarSystem', 'I-ResistanceStrength', 'I-PartsOfChemicalReactions', 'I-CelestialMovement', 'I-TypesOfChemicalReactions', 'B-Create', 'I-Circuits', 'I-GranularSolids', 'I-PropertiesOfSoil', 'I-ManmadeLocations', 'B-Fungi', 'B-ChangeInto', 'I-ExcretorySystem', 'B-Aquatic', 'I-AbilityAvailability', 'B-AirVehicle', 'B-Color', 'B-Scientists', 'B-Speciation', 'I-Eukaryote', 'I-Comet', 'B-MagneticForce', 'B-RespirationActions', 'I-LiquidMovement', 'I-InheritedBehavior', 'I-Monera', 'B-Associate', 'I-PropertiesOfFood', 'I-Occur', 'B-DensityUnit', 'B-UnderwaterEcosystem', 'B-Numbers', 'B-GeometricMeasurements', 'I-PerformingResearch', 'I-Rigidity', 'B-StarLayers', 'B-Magnetic', 'I-MolecularProperties', 'I-DURATION', 'B-Force', 'I-AmountComparison', 'I-RelativeDirection', 'I-ScientificTheoryExperimentationAndHistory', 'B-PartsOfABuilding', 'I-Uptake', 'B-AreaUnit', 'I-SoundMeasuringTools', 'I-Evolution', 'I-Gravity', 'B-Moon', 'I-ObjectQuantification', 'B-ScientificAssociationsAdministrations', 'I-TheoryOfMatter', 'I-PrenatalOrganismStates', 'B-CastFossilMoldFossil', 'I-TemperatureUnit', 'B-PushingForces', 'I-Consumption', 'B-ObservationTechniques', 'I-MoneyTerms', 'I-Representation', 'I-AnalyzingResearch', 'B-Behaviors', 'B-PartsOfASolution', 'B-ScientificMethod', 'I-PlanetParts', 'B-Constellation', 'B-SoundMeasuringTools', 'B-Bryophyte', 'I-CelestialObject', 'B-ExcretorySystem', 'B-SpaceProbes', 'I-BehavioralAdaptation', 'I-TerrestrialLocations', 'I-DATE', 'I-FossilRecordTimeline', 'B-Light', 'I-DistanceUnit', 'I-Reptile', 'B-MedicalTerms', 'I-TypesOfWaterInBodiesOfWater', 'B-ProbabilityAndCertainty', 'I-MeasurementsForHeatChange', 'I-Hypothesizing', 'B-Reproduction', 'I-CombineAdd', 'I-ElectricalProperty', 'B-Divide', 'B-ChangesToResources', 'B-TimesOfDayDayNight', 'I-ObservationTechniques', 'B-PartsOfTheImmuneSystem', 'B-NervousSystem', 'I-Light', 'I-StateOfMatter', 'B-Bacteria', 'I-MeasuringSpeed', 'I-PerformingExperimentsWell', 'I-Geography', 'I-PlantProcesses', 'B-PhaseChanges', 'B-Length', 'B-NutritiveSubstancesForAnimalsOrPlants', 'I-BusinessIndustry', 'I-PartsOfEndocrineSystem', 'I-ChangeInComposition', 'I-Length', 'B-DistanceMeasuringTools', 'I-Cities', 'B-Speed', 'I-AtomComponents', 'I-StateOfBeing', 'B-Amphibian', 'B-ScientificTools', 'B-Products', 'I-StarTypes', 'I-Examine', 'B-CellProcesses', 'I-LOCATION', 'I-Preserve', 'B-CirculationActions', 'I-CardinalNumber', 'I-PrepositionalDirections', 'B-Distance', 'B-Galaxy', 'I-BirdAnimalPart', 'B-LocationChangingActions', 'I-RelativeLocations', 'B-Event', 'B-PropertiesOfSoil', 'B-WavePerception', 'B-EnergyUnit', 'I-Foods', 'B-Countries', 'I-SpacecraftHumanRated', 'B-OtherHumanProperties', 'B-VolumeUnit', 'I-IllnessPreventionCuring', 'B-PartsOfTheEye', 'B-CelestialMeasurements', 'I-ParticleMovement', 'I-TypesOfEvent', 'I-MagneticEnergy', 'I-Toxins', 'B-Archaea', 'B-GeologicalEonsErasPeriodsEpochsAges', 'B-CardinalNumber', 'B-Gravity', 'B-MineralProperties', 'I-Unit', 'I-Problem', 'B-PartsOfTheRespiratorySystem', 'B-Pressure', 'I-AnimalCellPart', 'B-AcidityUnit', 'B-NorthernHemisphereLocations', 'I-PhysicalChange', 'B-Senses', 'B-MechanicalMovement', 'I-Exemplar', 'I-TimesOfDayDayNight', 'B-VariablesControls', 'B-TechnologicalInstrument', 'I-Magnetic', 'I-Scientists', 'B-StarTypes', 'I-MagneticDirectionMeasuringTool', 'I-Touch', 'I-SpeedUnit', 'B-Months', 'I-SystemParts', 'I-PartsOfObservationInstruments', 'I-ArithmeticMeasure', 'I-Mixtures', 'B-FeedbackMechanism', 'B-GaseousMatter', 'B-TypeOfConsumer', 'B-Unit', 'B-OtherGeographicWords', 'B-HeatingAppliance', 'I-PartsOfARepresentation', 'B-ConstructiveDestructiveForces',
                'I-GeopoliticalLocations', "[CLS]", "[SEP]"]
    def _create_examples(self,lines,set_type):
        examples = []
        for i,(sentence,label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples

    def _create_arc_examples(self,lines,set_type):
        examples = []
        for i, (sentence,label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples

    def _create_plain_text_examples(self,lines,set_type):
        examples = []
        for i, (sentence,label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list,0)}
    features = []
    for (ex_index,example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        valid = []
        label_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0,1)
        label_mask.insert(0,1)
        label_ids.append(np.array([int(i==label_map["[CLS]"]) for i in range(len(label_list))]).tolist())
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                temp_one_hop = np.zeros(len(label_list))
                for item in labels[i]:
                    temp_one_hop[label_map[item]]=1
                label_ids.append(temp_one_hop.tolist())
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(np.array([int(i==label_map["[SEP]"]) for i in range(len(label_list))]).tolist())
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(np.zeros(len(label_list)).tolist())
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(np.zeros(len(label_list)).tolist())
            label_mask.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask))
    return features

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--plain_text",
                        default=None,
                        type=str,
                        help="The plain_text_data directory")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval_test",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_eval_ARCtest",
                        action='store_true',
                        help="Whether to run eval on the ARC test set.")
    parser.add_argument("--do_eval_plain_text",
                        action='store_true',
                        help="Whether to run eval on the plain text data.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--group_length",
                        default=1000,
                        type=int,
                        help="In order to avoid the out of cpu memory, if your number of sentences is greater than this, divide it into groups")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--save_epochs', type=int, default=20,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {"ner":NerProcessor}

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = 0
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Prepare model
    config = BertConfig.from_pretrained(args.bert_model, num_labels=num_labels, finetuning_task=args.task_name)
    model = Ner.from_pretrained(args.bert_model,
              from_tf = False,
              config = config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    label_map = {i : label for i, label in enumerate(label_list,0)}
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        #print('all_label_ids shape: ', all_label_ids.shape)
        all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_valid_ids,all_lmask_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for epoch_flag in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, valid_ids,l_mask = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids,valid_ids,l_mask)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

            if args.local_rank in [-1, 0] and args.save_epochs > 0 and (epoch_flag+1) % args.save_epochs == 0:
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(epoch_flag+1))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model,'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                label_map = {i: label for i, label in enumerate(label_list, 0)}
                model_config = {"bert_model": args.bert_model, "do_lower": args.do_lower_case,
                                "max_seq_length": args.max_seq_length, "num_labels": len(label_list),
                                "label_map": label_map}
                json.dump(model_config, open(os.path.join(output_dir, "model_config.json"), "w"))

        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        label_map = {i : label for i, label in enumerate(label_list,0)}
        model_config = {"bert_model":args.bert_model,"do_lower":args.do_lower_case,"max_seq_length":args.max_seq_length,"num_labels":len(label_list),"label_map":label_map}
        json.dump(model_config,open(os.path.join(args.output_dir,"model_config.json"),"w"))

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            print('checkpoint： ', checkpoint)
            checkpoint_epoch = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = Ner.from_pretrained(checkpoint)
            model.to(device)
            model.eval()
            if args.do_eval_ARCtest:
                eval_examples = processor.get_ARC_test_examples(args.data_dir)
                eval_examples_length = math.ceil(len(eval_examples)/args.group_length)
                all_eval_examples = []
                for idx in range(eval_examples_length):
                    if (idx+1) == eval_examples_length:
                        all_eval_examples.append(eval_examples[int(idx*args.group_length):])
                    else:
                        all_eval_examples.append(eval_examples[int(idx*args.group_length):int((idx+1)*args.group_length)])
            elif args.do_eval_plain_text:
                eval_examples = processor.get_plain_text_examples(args.data_dir, args.plain_text)
                eval_examples_length = math.ceil(len(eval_examples) / args.group_length)
                all_eval_examples = []
                for idx in range(eval_examples_length):
                    if (idx + 1) == eval_examples_length:
                        all_eval_examples.append(eval_examples[int(idx * args.group_length):])
                    else:
                        all_eval_examples.append(eval_examples[int(idx * args.group_length):int((idx + 1) * args.group_length)])
            elif args.do_eval_test:
                eval_examples = processor.get_test_examples(args.data_dir)
                all_eval_examples = [eval_examples]
            else:
                eval_examples = processor.get_dev_examples(args.data_dir)
                all_eval_examples = [eval_examples]
            all_y_true = []
            all_y_pred = []
            for single_eval_examples in all_eval_examples:
                eval_features = convert_examples_to_features(single_eval_examples, label_list, args.max_seq_length, tokenizer)
                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(single_eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)
                all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
                all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
                all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
                eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids,
                                          all_lmask_ids)
                # Run prediction for full data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0
                y_true = []
                y_pred = []
                label_map = {i: label for i, label in enumerate(label_list, 0)}
                for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask in tqdm(eval_dataloader,
                                                                                             desc="Evaluating"):
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    valid_ids = valid_ids.to(device)
                    label_ids = label_ids.to(device)
                    l_mask = l_mask.to(device)

                    with torch.no_grad():
                        logits = model(input_ids, segment_ids, input_mask, valid_ids=valid_ids, attention_mask_label=l_mask)

                    logits = F.sigmoid(logits)
                    logits = logits.detach().cpu().numpy()
                    final_logits = (logits >= 0.4).astype(int)
                    label_ids = label_ids.to('cpu').numpy()
                    input_mask = input_mask.to('cpu').numpy()

                    for i, label in enumerate(label_ids):
                        temp_1 = []
                        temp_2 = []
                        flag = 0
                        for j, m in enumerate(label):
                            if j == 0:
                                continue
                            elif flag == len(label_map) - 1:
                                y_true.append(temp_1)
                                y_pred.append(temp_2)
                                break
                            else:
                                temp_label_ids_list = []
                                temp_logits_list = []
                                for label_temp_idx, label_temp in enumerate(label_ids[i][j]):
                                    if int(label_temp) == 1:
                                        flag = label_temp_idx
                                        if flag == len(label_map) - 1:
                                            break
                                        temp_label_ids_list.append(label_map[label_temp_idx])
                                for logit_temp_idx, logits_temp in enumerate(final_logits[i][j]):
                                    if flag == len(label_map) - 1:
                                        break
                                    if int(logits_temp) == 1:
                                        temp_logits_list.append(label_map[logit_temp_idx])
                                if flag != len(label_map) - 1:
                                    temp_1.append(temp_label_ids_list)
                                    temp_2.append(temp_logits_list)
                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)
            if args.do_eval_ARCtest:
                token_predition_write(os.path.join(args.data_dir, "ARC_test_spacy.txt"), all_y_pred, checkpoint, 'ARC_test')
            elif args.do_eval_plain_text:
                base_filename = os.path.basename(args.plain_text)
                token_predition_write(os.path.join(args.data_dir, 'conll_'+base_filename), all_y_pred, checkpoint,
                                      base_filename)
            elif args.do_eval_test:
                report = classification_report(all_y_true, all_y_pred, digits=4)
                logger.info("\n%s", report)
                output_eval_file = os.path.join(checkpoint, "test_results.txt")
                with open(output_eval_file, "w") as writer:
                    logger.info("***** test results *****")
                    logger.info("\n%s", report)
                    writer.write(report)
            else:
                report = classification_report(all_y_true, all_y_pred, digits=4)
                logger.info("\n%s", report)
                output_eval_file = os.path.join(checkpoint, "eval_results.txt")
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results *****")
                    logger.info("\n%s", report)
                    writer.write(report)

if __name__ == "__main__":
        main()
