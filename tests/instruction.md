How to Run the Tests

Install unittest:

    The unittest framework is part of Python's standard library, so no additional installation is required.

Run the Tests:

    N avigate to the tests/ folder and run the tests using the following commands:

    python -m unittest test_classify.py
    python -m unittest test_model.py
    python -m unittest test_update.py

Notes
Test Data:

    The tests create sample PDFs and CSV files in the ../data/test/ directory. These files are automatically cleaned up after the tests run.

Dependencies:

    Ensure that the scripts/ and utils/ modules are correctly imported in the test files.

Customization:

    Update paths and configurations in the test files to match your project structure.