import os
import json
from datasets import DatasetInfo, GeneratorBasedBuilder, SplitGenerator, Split, Features, Value, Image as HFImage

class MyControlNetDataset(GeneratorBasedBuilder):
    def _info(self):
        return DatasetInfo(
            description="Custom ControlNet dataset with conditioning image, target image and prompt",
            features=Features({
                "image": HFImage(),
                "conditioning_image": HFImage(),
                "prompt": Value("string")
            }),
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(self.config.data_dir)
        return [
            SplitGenerator(name=Split.TRAIN, gen_kwargs={"data_dir": data_dir}),
        ]

    def _generate_examples(self, data_dir):
        metadata_path = os.path.join(data_dir, "dataset.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        for idx, sample in enumerate(metadata):
            yield idx, {
                "image": os.path.join(data_dir, sample["image"]),
                "conditioning_image": os.path.join(data_dir, sample["conditioning_image"]),
                "prompt": sample["prompt"]
            }