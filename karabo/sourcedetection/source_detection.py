import numpy as np
from scipy.spatial.distance import cdist

from karabo.Imaging.image import Image, open_fits_image
from karabo.simulation.sky_model import SkyModel
from karabo.sourcedetection.evaluation import SourceDetectionEvaluation
from karabo.sourcedetection.result import SourceDetectionResult, PyBDSFSourceDetectionResult
from karabo.util.data_util import read_CSV_to_ndarray


def read_detection_from_sources_file_csv(filepath: str, source_image_path: str = None) -> SourceDetectionResult:
    """
    Reads in a CSV table and saves it in the Source Detection Result.
    The format of the CSV is according to the PyBDSF definition.:
    https://www.astron.nl/citt/pybdsf/write_catalog.html#definition-of-output-columns

    Karabo creates the output from write_catalog(format='csv', catalogue_type='gaul').
    We suggest to only read in CSV that are created with Karabo (or with PyBDSF itself with the above configuration).

    This method is mainly for convenience.
    It allows that one can save the CSV with the SourceDetectionResult.save_sources_as_csv_file("./sources.csv")
    and then read it back in.
    This helps save runtime and potential wait time, when working with the output of the source detection

    :param source_image_path: (Optional), you can also read in the source image for the detection.
            If you read this back in you can use plot() function on the SkyModelToSourceDetectionMapping
    :param filepath: file of CSV sources in the format that
    :return: SourceDetectionResult
    """
    image = None
    if source_image_path is not None:
        image = open_fits_image(source_image_path)
    detected_sources = read_CSV_to_ndarray(filepath)
    detection = SourceDetectionResult(detected_sources=detected_sources, source_image=image)
    return detection


def detect_sources_in_image(image: Image, beam=None) -> SourceDetectionResult:
    """
    Detecting sources in an image. The Source detection is impemented with the PyBDSF.process_image function.
    See https://www.astron.nl/citt/pybdsf/process_image.html for more information.

    :param image: Image to perform source detection on
    :param beam: FWHM of restoring beam. Specify as (maj, min. pos angle E of N).
                 None means it will try to be extracted from the Image data. (Might fail)
    :return: Source Detection Result containing the found sources
    """
    import bdsf
    detection = bdsf.process_image(image.file.path, beam=beam, quiet=True, format='csv')
    return PyBDSFSourceDetectionResult(detection)


def evaluate_result_with_sky(source_detection_result: SourceDetectionResult, sky: SkyModel,
                             source_image_cell_size: float, distance_threshold: float):
    """
    Evaluate Result of a Source Detection Result by comparing it with the original sky (in Pixel space).
    The mapping uses the automatic_assignment_of_ground_truth_and_prediction() function
    and calculate_evaluation_measures() to create the evaluation

    :param source_detection_result: result that was produced with a source detection algorithm
    :param sky: sky that was used to create the image
    :param source_image_cell_size: cellsize in the original source image (used for mapping),
                                   cannot be read out from the fits file (unfortunately)
    :param distance_threshold: threshold of distance between two sources,
                               so that they are still considered in mathching (pixel distance).
    :return:
    """
    image = source_detection_result.get_source_image()
    sky_projection_pixel_per_side = image.get_dimensions_of_image()[0]

    truth = sky.project_sky_to_2d_image(source_image_cell_size, sky_projection_pixel_per_side)[:2].astype(
        'float64')
    pred = np.array(source_detection_result.get_pixel_position_of_sources()).astype('float64')
    assignment = automatic_assignment_of_ground_truth_and_prediction(truth, pred, distance_threshold)
    tp, fp, fn = calculate_evaluation_measures(assignment, truth, pred)
    result = SourceDetectionEvaluation(assignment, truth, sky, pred, source_detection_result, tp, fp, fn)
    return result


def automatic_assignment_of_ground_truth_and_prediction(ground_truth: np.ndarray, detected: np.ndarray,
                                                        max_dist: float) -> np.ndarray:
    """
    Automatic assignment of the predicted sources `predicted` to the ground truth `gtruth`.
    The strategy is the following
    (similar to AUTOMATIC SOURCE DETECTION IN ASTRONOMICAL IMAGES, P.61, Marc MASIAS MOYSET, 2014):

    Each distance between the predicted and the ground truth sources is calculated.
    Any distances > `max_dist` are deleted.
    Assign the closest distance from the predicted and ground truth.
    Repeat the assignment, until every source from the gtruth has an assigment if possible,
        not allowing any double assignments from the predicted sources to the ground truth and vice versa.
    So each ground truth source should be assigned with a predicted source if at leas one was in range
        and the predicted source assigned to another ground truth source before.

    :param ground_truth: nx2 np.ndarray with the ground truth pixel coordinates of the catalog
    :param detected: kx2 np.ndarray with the predicted pixel coordinates of the image
    :param max_dist: maximal allowed distance for assignment (in pixel)

    :return: jx3 np.ndarray where each row represents an assignment
                 - first column represents the ground truth index
                 - second column represents the predicted index
                 - third column represents the euclidean distance between the assignment
    """
    ground_truth = ground_truth.transpose()
    detected = detected.transpose()
    euclidian_distances = cdist(ground_truth, detected)
    ground_truth_assignments = np.array([None] * ground_truth.shape[0])
    # gets the euclidian_distances sorted values indices as (m*n of euclidian_distances) x 2 matrix
    argsort_2dIndexes = np.array(
        np.unravel_index(np.argsort(euclidian_distances, axis=None), euclidian_distances.shape)).transpose()
    max_dist_2dIndexes = np.array(np.where(euclidian_distances <= max_dist)).transpose()
    # can slice it since argsort_2dIndexes is sorted. it is to ensure to not assign sources outside of max_dist
    argsort_2dIndexes = argsort_2dIndexes[:max_dist_2dIndexes.shape[0]]
    # to get the closes assignment it is the task to get the first indices pair which each index in each column
    # occured just once
    assigned_ground_truth_indexes, assigned_predicted_idxs, eucl_dist = [], [], []
    for i in range(argsort_2dIndexes.shape[0]):
        # could maybe perform better if possible assignments argsort_2dIndexes is very large by filtering the
        # selected idxs after assignment
        assignment_idxs = argsort_2dIndexes[i]
        if (assignment_idxs[0] not in assigned_ground_truth_indexes) and (
                assignment_idxs[1] not in assigned_predicted_idxs):
            assigned_ground_truth_indexes.append(assignment_idxs[0])
            assigned_predicted_idxs.append(assignment_idxs[1])
            eucl_dist.append(euclidian_distances[assignment_idxs[0], assignment_idxs[1]])
    assignments = np.array([assigned_ground_truth_indexes, assigned_predicted_idxs, eucl_dist]).transpose()
    return assignments


def calculate_evaluation_measures(assignments: np.ndarray, ground_truth: np.ndarray,
                                  detected: np.ndarray) -> tuple:
    """
    Calculates the True Positive (TP), False Positive (FP) and False Negative (FN) of the ground truth and predictions.
    - TP are the detections associated with a source
    - FP are detections without any associated source
    - FN are sources with no associations with a detection

    :param assignments:
    :param ground_truth: nx2 np.ndarray with the ground truth pixel coordinates of the catalog
    :param detected: kx2 np.ndarray with the predicted pixel coordinates of the image
    :param max_dist: maximal allowed distance for assignment

    :return: TP, FP, FN
    """
    tp = assignments.shape[0]
    fp = detected.shape[0] - assignments.shape[0]
    fn = ground_truth.shape[0] - assignments.shape[0]
    return tp, fp, fn
