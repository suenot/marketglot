from book.book import LocalBook


def make_book():
    book = LocalBook()
    book.apply_snapshot(
        prices=[100.0, 99.0, 98.0, 101.0, 102.0, 103.0],
        qtys=[1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
        sides=["bid", "bid", "bid", "ask", "ask", "ask"],
    )
    return book


def test_snapshot_populates_both_sides():
    book = make_book()
    assert book.bids == {100.0: 1.0, 99.0: 2.0, 98.0: 3.0}
    assert book.asks == {101.0: 1.5, 102.0: 2.5, 103.0: 3.5}


def test_snapshot_replaces_state():
    book = make_book()
    book.apply_snapshot(prices=[50.0], qtys=[1.0], sides=["bid"])
    assert book.bids == {50.0: 1.0}
    assert book.asks == {}


def test_delta_updates_existing_level():
    book = make_book()
    book.apply_delta(prices=[100.0], qtys=[5.0], sides=["bid"])
    assert book.bids[100.0] == 5.0


def test_delta_adds_new_level():
    book = make_book()
    book.apply_delta(prices=[97.0], qtys=[4.0], sides=["bid"])
    assert book.bids[97.0] == 4.0


def test_delta_zero_qty_removes_level():
    book = make_book()
    book.apply_delta(prices=[100.0], qtys=[0.0], sides=["bid"])
    assert 100.0 not in book.bids
    # Removing a missing level is a no-op, not an error.
    book.apply_delta(prices=[12345.0], qtys=[0.0], sides=["bid"])


def test_top_levels_ordering_and_depth():
    book = make_book()
    bids, asks = book.top_levels(depth=2)
    assert bids == [(100.0, 1.0), (99.0, 2.0)]
    assert asks == [(101.0, 1.5), (102.0, 2.5)]


def test_top_levels_depth_exceeds_book():
    book = make_book()
    bids, asks = book.top_levels(depth=10)
    assert len(bids) == 3
    assert len(asks) == 3


def test_mid():
    book = make_book()
    assert book.mid() == (100.0 + 101.0) / 2.0


def test_mid_none_when_side_empty():
    book = LocalBook()
    book.apply_snapshot(prices=[100.0], qtys=[1.0], sides=["bid"])
    assert book.mid() is None


def test_is_valid_normal_book():
    assert make_book().is_valid() is True


def test_is_valid_empty_book():
    assert LocalBook().is_valid() is False


def test_is_valid_one_sided():
    book = LocalBook()
    book.apply_snapshot(prices=[100.0], qtys=[1.0], sides=["bid"])
    assert book.is_valid() is False


def test_is_valid_crossed_book():
    book = LocalBook()
    book.apply_snapshot(
        prices=[105.0, 100.0],
        qtys=[1.0, 1.0],
        sides=["bid", "ask"],
    )
    # best_bid 105 > best_ask 100 -> crossed -> invalid.
    assert book.is_valid() is False
    assert book.bids and book.asks
