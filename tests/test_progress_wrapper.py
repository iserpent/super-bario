from super_bario import progress, Progress, Theme

def test_progress_wrapper_yields_items():
    items = list(range(5))
    seen = []
    for item in progress(items, title="Test", theme=Theme.minimal()):
        seen.append(item)
    assert seen == items
    Progress.close()

def test_group_smoke():
    group = Progress.instance()
    bar = group.create_bar(total=3, title="Smoke")
    for _ in range(3):
        bar.increment()
        group.display(force_update=True, force_clear=True)
    group.close()
