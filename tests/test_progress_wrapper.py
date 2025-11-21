from super_bario import progress, Group, Theme

def test_progress_wrapper_yields_items():
    items = list(range(5))
    seen = []
    for item in progress(items, title="Test", theme=Theme.minimal()):
        seen.append(item)
    assert seen == items
    Group.close()

def test_group_smoke():
    group = Group.instance()
    bar = group.create_bar(total=3, title="Smoke")
    for _ in range(3):
        bar.increment()
        group.display(force_update=True, force_clear=True)
    group.close()
