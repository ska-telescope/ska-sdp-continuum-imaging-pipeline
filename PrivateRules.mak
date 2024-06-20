#
# Custom python rules for developers
#

# Same as python-lint, but do not build a code analysis report
python-dev-lint:
	isort --check-only --profile black --line-length $(PYTHON_LINE_LENGTH) $(PYTHON_SWITCHES_FOR_ISORT) $(PYTHON_LINT_TARGET)
	black --exclude .+\.ipynb --check --line-length $(PYTHON_LINE_LENGTH) $(PYTHON_SWITCHES_FOR_ISORT) $(PYTHON_LINT_TARGET)
	flake8 --show-source --statistics --max-line-length $(PYTHON_LINE_LENGTH) $(PYTHON_LINT_TARGET)
	pylint --max-line-length $(PYTHON_LINE_LENGTH) $(PYTHON_LINT_TARGET)

# Same as python-test, but do not create a `build` directory, do not to create
# and attempt to upload a coverage report
python-dev-test:
	pytest -vv --cov=src --cov-report=term-missing tests/

# Run all code checks necessary to pass the CI pipeline
python-dev-checks: python-dev-lint python-dev-test

.PHONY: python-dev-lint python-dev-test python-dev-checks

