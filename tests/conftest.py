import os
import vcr
from dotenv import load_dotenv

load_dotenv()

if os.getenv("OPENAI_API_KEY") is None:
    raise ValueError(
        "OPENAI_API_KEY environment variable is not set. Please set it to run the tests."
    )

my_vcr = vcr.VCR(
    filter_headers=[("authorization", "DUMMY")],
    path_transformer=vcr.VCR.ensure_suffix(".yaml"),
    cassette_library_dir="tests/cassettes",
)


def pytest_configure(config):
    config.option.vcr_record_mode = "once"
