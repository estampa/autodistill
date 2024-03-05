import datetime
import enum
import glob
import os
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import roboflow
import supervision as sv
from PIL import Image
from supervision.utils.file import save_text_file
from tqdm import tqdm

from autodistill.core import BaseModel
from autodistill.helpers import load_image, split_data

from .detection_ontology import DetectionOntology
from ..core import Ontology


class NmsSetting(str, enum.Enum):
    NONE = "no_nms"
    CLASS_SPECIFIC = "class_specific"
    CLASS_AGNOSTIC = "class_agnostic"


@dataclass
class DetectionBaseModel(BaseModel):
    ontology: DetectionOntology

    def __init__(self, ontology: Ontology):
        super().__init__(ontology=ontology)

        self._annotator = sv.BoundingBoxAnnotator()
        self._label_annotator = sv.LabelAnnotator(background=False, shadow=True)

    @abstractmethod
    def predict(self, input: str | np.ndarray | Image.Image) -> sv.Detections:
        pass

    def sahi_predict(self, input: str | np.ndarray | Image.Image) -> sv.Detections:
        slicer = sv.InferenceSlicer(callback=self.predict)

        return slicer(load_image(input, return_format="cv2"))

    def _record_confidence_in_files(
        self,
        annotations_directory_path: str,
        images: Dict[str, np.ndarray],
        annotations: Dict[str, sv.Detections],
    ) -> None:
        Path(annotations_directory_path).mkdir(parents=True, exist_ok=True)
        for image_name, _ in images.items():
            detections = annotations[image_name]
            yolo_annotations_name, _ = os.path.splitext(image_name)
            confidence_path = os.path.join(
                annotations_directory_path,
                "confidence-" + yolo_annotations_name + ".txt",
            )
            confidence_list = [str(x) for x in detections.confidence.tolist()]
            save_text_file(lines=confidence_list, file_path=confidence_path)
            print("Saved confidence file: " + confidence_path)

    def _anotate_image(self, image, detections, classes):
        annotated_frame = self._annotator.annotate(scene=image.copy(), detections=detections)

        labels = [
            f"{classes[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _ in detections
        ]
        annotated_frame = self._label_annotator.annotate(
            scene=annotated_frame, labels=labels, detections=detections
        )

        return annotated_frame

    def _save_dataset(
        self,
        images_map: Dict[str, np.ndarray],
        detections_map: Dict[str, sv.Detections],
        output_folder: str,
        record_confidence: bool,
        save_predictions: bool
    ) -> None:
        dataset = sv.DetectionDataset(
            self.ontology.classes(), images_map, detections_map
        )

        dataset.as_yolo(
            output_folder + "/images",
            output_folder + "/annotations",
            min_image_area_percentage=0.01,
            data_yaml_path=output_folder + "/data.yaml",
        )

        if save_predictions:
            with sv.ImageSink(target_dir_path=output_folder + "/detections",
                              overwrite=False) as sink:
                for image_name, image_data in images_map.items():
                    detection = detections_map[image_name]
                    output_image = self._anotate_image(image_data, detection, self.ontology.classes())

                    sink.save_image(image=output_image, image_name=image_name)

        if record_confidence is True:
            self._record_confidence_in_files(
                output_folder + "/annotations", images_map, detections_map
            )
        split_data(output_folder, record_confidence=record_confidence)

    @staticmethod
    def _get_processed_images(output_folder: str) -> [str]:
        processed_images = []
        for root, dirs, files in os.walk(output_folder):
            for file in files:
                if file.endswith(".jpg"):
                    processed_images.append(file)

        return processed_images

    def label(
        self,
        input_folder: str,
        extension: str = ".jpg",
        output_folder: str = None,
        human_in_the_loop: bool = False,
        roboflow_project: str = None,
        roboflow_tags: str = ["autodistill"],
        sahi: bool = False,
        record_confidence: bool = False,
        nms_settings: NmsSetting = NmsSetting.NONE,
        batch_save: int = None,
        save_predictions: bool = False,
    ):  # -> sv.DetectionDataset:
        """
        Label a dataset with the model.
        """
        if output_folder is None:
            output_folder = input_folder + "_labeled"

        os.makedirs(output_folder, exist_ok=True)

        processed_images = self._get_processed_images(output_folder)

        images_map = {}
        detections_map = {}

        if sahi:
            slicer = sv.InferenceSlicer(callback=self.predict)

        files = glob.glob(input_folder + "/*" + extension)
        progress_bar = tqdm(files, desc="Labeling images")
        # iterate through images in input_folder
        for f_path in progress_bar:
            if os.path.basename(f_path) in processed_images:
                continue

            progress_bar.set_description(desc=f"Labeling {f_path}", refresh=True)
            image = cv2.imread(f_path)

            f_path_short = os.path.basename(f_path)
            images_map[f_path_short] = image.copy()

            if sahi:
                detections = slicer(image)
            else:
                detections = self.predict(image)

            if nms_settings == NmsSetting.CLASS_SPECIFIC:
                detections = detections.with_nms()
            if nms_settings == NmsSetting.CLASS_AGNOSTIC:
                detections = detections.with_nms(class_agnostic=True)

            detections_map[f_path_short] = detections

            if batch_save is not None and (len(images_map) + 1) % batch_save == 0:
                self._save_dataset(
                    images_map, detections_map, output_folder, record_confidence, save_predictions
                )
                images_map = {}
                detections_map = {}

        self._save_dataset(images_map, detections_map, output_folder, record_confidence, save_predictions)

        if human_in_the_loop:
            roboflow.login()

            rf = roboflow.Roboflow()

            workspace = rf.workspace()

            workspace.upload_dataset(output_folder, project_name=roboflow_project)

        print("Labeled dataset created - ready for distillation.")
        # return dataset
