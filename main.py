import logging

from src.pipeline.training_pipeline import TrainingPipeline


if __name__ == "__main__":
    try:
        TrainingPipeline().run_pipeline()
    except Exception:
        logging.exception("Fatal training pipeline error")
        raise
