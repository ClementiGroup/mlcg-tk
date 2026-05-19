# Tests for `mlcg-tk` Package

The `mlcg-tk` package includes a structured test suite to ensure code quality and correctness. Tests are organized by **type**:

- **Unit tests**  
  Located in `tests/unit/`.  
  These tests are fast and test individual functions or small components in isolation.

- **Integration tests**  
  Located in `tests/integration/`.  
  These tests check the interaction between multiple components and may involve longer-running pipelines or CLI commands.

---

## Running Tests

### Run all tests
```bash
pytest
```
### Run only integration tests
```bash
pytest -m integration
```
### Run only unit tests
```bash
pytest -m units
```
## Adding New Tests

When contributing new tests:

- Place unit tests in tests/unit/.
- Place integration tests in tests/integration/.
- Markers for integration and unit are automatically added to each test, so it is not necessary to explicitely add them.

This ensures tests are correctly categorized and can be run selectively.