import time
from super_bario import Bar

def test_bar_initial_state():
    bar = Bar(total=10)
    assert bar.total == 10
    assert bar.current == 0
    assert not bar.is_complete()

def test_bar_increment():
    bar = Bar(total=2)
    bar.increment()
    assert bar.current == 1
    bar.increment()
    assert bar.current == 2
    assert bar.is_complete()

def test_elapsed_and_eta():
    bar = Bar(total=5)
    with bar:
        bar.increment()
        time.sleep(0.01)
        assert isinstance(bar.elapsed_time(), float)
        eta = bar.estimated_time()
        if eta is not None:
            assert isinstance(eta, float)
