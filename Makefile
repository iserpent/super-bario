publish: test build
	@echo "🚀 Publishing the package"
	uv run twine upload --repository pypi dist/*

.PHONY: build
build: clean
	@echo "🏗️ Building the package"
	uv build

.PHONY: test
test:
	@echo "🧪 Running tests"
	uv run pytest -v

.PHONY: clean
clean:
	@echo "🧹 Cleaning build artifacts"
	rm -rf dist src/*.egg-info
