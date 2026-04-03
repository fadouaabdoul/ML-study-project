import logging 

logging.basicConfig(
    filename="ML.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("start of debugging")