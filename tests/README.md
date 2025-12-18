# Tests Directory

## Planned Test Coverage

### Unit Tests

- [ ] `test_time_series.py` - Time series model functions
- [ ] `test_data_loading.py` - Data loading and validation
- [ ] `test_statistics.py` - Statistical utilities (future)

### Integration Tests

- [ ] `test_dashboard.py` - Streamlit dashboard components
- [ ] `test_ml_pipeline.py` - ML model training and evaluation

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=scripts --cov-report=html
```

## Test Data

Test fixtures are stored in `tests/fixtures/` (to be created).

---
*Tests to be implemented as part of Phase 2 improvements*
