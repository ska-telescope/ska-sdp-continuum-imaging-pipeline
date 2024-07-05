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
	rm -f .coverage
	rm -f .coverage.*
	pytest -vv --cov=src --cov-report= tests/
	coverage combine -a
	coverage report --show-missing

# Run all code checks necessary to pass the CI pipeline
python-dev-checks: python-dev-lint python-dev-test

# Same as python-do-test, but make sure we get accurate coverage for code
# being run only on dask workers. We just need to combine the multiple
# coverage reports that are generated with a call to  `coverage combine -a`.
python-do-test-with-dask-coverage:
	@$(PYTHON_RUNNER) pytest --version -c /dev/null
	@mkdir -p build
	$(PYTHON_VARS_BEFORE_PYTEST) $(PYTHON_RUNNER) pytest $(PYTHON_VARS_AFTER_PYTEST) \
	 --cov=$(PYTHON_SRC) --cov-report= --junitxml=build/reports/unit-tests.xml
	coverage combine -a
	coverage report --show-missing
	coverage html --directory=build/reports/code-coverage
	coverage xml -o build/reports/code-coverage.xml

python-test-with-dask-coverage: python-pre-test python-do-test-with-dask-coverage python-post-test 

.PHONY: python-dev-lint python-dev-test python-dev-checks python-test-with-dask-coverage
