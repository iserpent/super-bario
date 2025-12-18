"""Examples demonstrating the new progress bar features with colors and themes"""

import sys
import time
import random

from super_bario import (
    progress,
    Progress,
    Bar,
    View,
    Theme,
    TitleWidget,
    BarWidget,
    PercentageWidget,
    ElapsedTimeWidget,
    EstimatedTimeWidget,
    CounterWidget,
    SpinnerWidget,
    RateWidget,
    Colors,
)


def example_0():
    print("=== Example 0: Different Themes in nested repeated layouts with Threads ===")

    import threading

    def worker(theme, layouts, indent, remove_on_complete=False):
        for i in progress(
            range(1, 80 + 1),
            title=lambda item: f"Task {layouts} - Item {item.value}",
            theme=theme,
            layouts=layouts,
            indent=indent,
            remove_on_complete=remove_on_complete,
        ):
            time.sleep(0.1)

    def custom_bar_worker(bar):
        for i in progress(range(1, 80 + 1), bar=bar):
            time.sleep(0.1)

    def sub_worker(layouts):
        for i in progress(
            range(1, 2 + 1),
            title=lambda item: f"Processing i: {item.value}",
            layouts=layouts,
        ):
            for j in progress(
                range(1, 10 + 1),
                title=lambda item: f"Processing j: {item.value}",
                indent=1,
                layouts=layouts,
                remove_on_complete=True,
            ):
                for k in progress(
                    range(1, 30 + 1),
                    title=lambda item: f"Processing k: {item.value}",
                    indent=2,
                    layouts=layouts,
                    remove_on_complete=True,
                ):
                    time.sleep(0.01)

    def stdout_worker():
        for i in range(1, 10 + 1):
            time.sleep(random.uniform(0.2, 0.5))
            print(f"Demo of the multi-threaded stdout handling: Step {i}")

    def stderr_worker():
        for i in range(1, 10 + 1):
            time.sleep(random.uniform(0.4, 0.5))
            print(
                f"Demo of the multi-threaded stderr handling: Step {i}", file=sys.stderr
            )

    # Use sys.stdout or sys.stderr as output stream for Progress (default is sys.stderr)
    Progress.stream = sys.stdout

    Progress.create_row("h_layout1")
    Progress.create_row("h_layout2")
    Progress.create_row("h_layout3")
    Progress.create_row("h_layout4")

    Progress.create_column("v_layout1_1", parents=["h_layout1"])
    Progress.create_column("v_layout2_1", parents=["h_layout1"])
    Progress.create_column("v_layout3_1", parents=["h_layout1"])

    Progress.create_column("v_layout1_2", parents=["h_layout2"])
    Progress.create_column("v_layout2_2", parents=["h_layout2"])
    Progress.create_column("v_layout3_2", parents=["h_layout2"])

    Progress.create_column("v_layout1_3", parents=["h_layout3"])
    Progress.create_column("v_layout2_3", parents=["h_layout3"])

    Progress.create_column("v_layout1_4", parents=["h_layout4"])
    Progress.create_column("v_layout2_4", parents=["h_layout4"])

    Progress.add_layout("h_layout2", parents=["v_layout1_4"])
    Progress.add_layout("h_layout3", parents=["v_layout2_4"])

    threads = []

    themes = [
        (Theme.default(), ["v_layout1_1"], 0, False),
        (Theme.matrix(), ["v_layout2_1"], 1, True),
        (Theme.fire(), ["v_layout3_1"], 2, True),
    ]

    for args in themes:
        t = threading.Thread(target=worker, args=args)
        threads.append(t)

    themes = [
        (Theme.default(), ["v_layout1_3"], 0, False),
        (Theme.minimal(), ["v_layout1_3"], 1, False),
        (Theme.matrix(), ["v_layout1_3"], 2, True),
        (Theme.fire(), ["v_layout1_3"], 3, True),
        (Theme.load(), ["v_layout1_3"], 4, True),
    ]
    for args in themes:
        t = threading.Thread(target=worker, args=args)
        threads.append(t)

    bar = Progress.add_custom_bar(
        total=500,
        title=f"Custom fractions",
        indent=0,
        remove_on_complete=False,
        layouts=["v_layout1_3"],
        char_start_incomplete="",
        char_end_incomplete="",
        char_incomplete=" ",
        char_complete="⣿",
        char_complete_fractions=["⣀", "⣄", "⣆", "⣇", "⣧", "⣷", "⣿"],
    )

    t = threading.Thread(target=custom_bar_worker, args=(bar,))
    threads.append(t)

    complete_chars = ["━", "▰", "▪", "⣿"]
    incomplete_chars = [" ", "▱", "▫", " "]
    for idx, theme in enumerate(
        [Theme.default(), Theme.minimal(), Theme.matrix(), Theme.fire(), Theme.load()]
    ):
        bar = Progress.add_custom_bar(
            total=80,
            title=f"Custom Bar {idx}",
            theme=theme,
            layouts=["v_layout1_3"],
            indent=0,
            remove_on_complete=False,
            char_start_incomplete="",
            char_end_incomplete="",
            char_incomplete=incomplete_chars[idx % len(incomplete_chars)],
            char_complete=complete_chars[idx % len(complete_chars)],
            char_complete_fractions=[],
        )
        t = threading.Thread(target=custom_bar_worker, args=(bar,))
        threads.append(t)

    threads.append(
        threading.Thread(
            target=sub_worker, args=(["v_layout1_2", "v_layout2_2", "v_layout3_2"],)
        )
    )
    threads.append(threading.Thread(target=sub_worker, args=(["v_layout2_3"],)))
    threads.append(threading.Thread(target=stdout_worker))
    threads.append(threading.Thread(target=stderr_worker))

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    Progress.close()


def example_1():
    print("=== Example 1: Default Theme ===")

    for i in progress(range(1, 100 + 1), title="Processing items"):
        time.sleep(0.02)

    # We need to close the Progress only if we need to create a new one later, like in this script
    Progress.close()


def example_2():
    print("=== Example 2: Custom Icons ===")

    # Enable force redraw to properly display custom icons: unicode characters may have different widths,
    # and soft redraws may not handle them correctly.
    Progress.force_redraw = True

    bars = []
    bar = Progress.add_custom_bar(
        total=80,
        title="Archery",
        indent=0,
        remove_on_complete=False,
        char_start_incomplete="🏹",
        char_start_complete="🏅",
        char_end_incomplete="",
        char_end_complete="🎯",
        char_incomplete=" ",
        char_complete=" ",
        char_complete_fractions=["➳"],
    )

    bars.append(bar)

    bar = Progress.add_custom_bar(
        total=80,
        title="Soccer",
        indent=0,
        remove_on_complete=False,
        char_start_incomplete="🏃",
        char_end_incomplete="🥅",
        char_end_complete="🏆",
        char_incomplete=" ",
        char_complete=" ",
        char_complete_fractions=["⚽"],
    )

    bars.append(bar)

    for i in range(1, 80 + 1):
        for bar in bars:
            bar.increment()
        Progress.display()
        time.sleep(0.05)

    Progress.close()


def example_3():
    print("=== Example 3: Watch Qeueus ===")
    from queue import Queue

    q1 = Queue(maxsize=1000)
    q2 = Queue(maxsize=1000)

    for i in range(1, 701 + 1):
        q1.put_nowait(i)

    for i in range(1, 301 + 1):
        q2.put_nowait(i)

    Progress.watch_interval = 0.3

    Progress.create_row("h_layout1")

    Progress.create_column("v_layout1", parents=["h_layout1"])
    Progress.create_column("v_layout2", parents=["h_layout1"])

    Progress.add_watch(q1, "Queue 1", layouts=["v_layout1"])
    Progress.add_watch(q2, "Queue 2", layouts=["v_layout2"])

    for i in range(1, 200 + 1):
        q1_action = random.choice([q1.get_nowait, q1.put_nowait])
        q2_action = random.choice([q2.get_nowait, q2.put_nowait])

        for j in range(1, random.randint(1, 50) + 1):
            if q1_action == q1.put_nowait and not q1.full():
                q1.put_nowait(i)
            elif q1_action == q1.get_nowait and not q1.empty():
                q1.get_nowait()

        for k in range(1, random.randint(1, 50) + 1):
            if q2_action == q2.put_nowait and not q2.full():
                q2.put_nowait(i)
            elif q2_action == q2.get_nowait and not q2.empty():
                q2.get_nowait()

        time.sleep(0.02)

    Progress.close()


def example_4():
    print("=== Example 4: Minimal Theme ===")

    with Progress:  # another way to manage Progress lifecycle
        for i in progress(
            range(1, 100 + 1), title="Processing items", theme=Theme.minimal()
        ):
            time.sleep(0.02)


def example_5():
    print("=== Example 5: Text only ===")

    with Progress:
        for i in progress(
            range(1, 100 + 1), title="Processing items", use_unicode=False
        ):
            time.sleep(0.02)


def example_6():
    print("=== Example 6: Matrix Theme ===")

    with Progress:
        for i in progress(
            range(1, 100 + 1), title="Processing items", theme=Theme.matrix()
        ):
            time.sleep(0.02)


def example_7():
    print("=== Example 7: Fire Gradient ===")

    bar = Bar(total=100, title="Heating up")
    view = View(bar, theme=Theme.fire())
    Progress.add_bar(bar, view)

    for i in range(1, 100 + 1):
        bar.increment()
        Progress.display()
        time.sleep(0.02)

    Progress.close()


def example_8():
    print("=== Example 8: Load Gradient ===")

    bar = Bar(total=100, title="Load")
    view = View(bar, theme=Theme.load())
    Progress.add_bar(bar, view)

    for i in range(1, 100 + 1):
        bar.increment()
        Progress.display()
        time.sleep(0.02)

    Progress.close()


def example_9():
    print("=== Example 9: Custom theme ===")

    # Custom gradient theme
    custom_gradient = Theme(
        title_color=Colors.BRIGHT_YELLOW,
        use_gradient=True,
        gradient_start=(255, 0, 128),  # Pink
        gradient_end=(128, 0, 255),  # Purple
        percentage_color=Colors.BRIGHT_MAGENTA,
        time_color=Colors.BRIGHT_CYAN,
    )

    bar = Bar(total=100, title="Rainbow Progress")
    view = View(bar, theme=custom_gradient)
    Progress.add_bar(bar, view)

    for i in range(1, 100 + 1):
        bar.increment()
        Progress.display()
        time.sleep(0.02)

    Progress.close()


def example_10():
    print("=== Example 10: Custom Widgets with Rate ===")

    widgets = [
        TitleWidget("Download", theme=Theme.default()),
        BarWidget(theme=Theme.default()),
        PercentageWidget(theme=Theme.default()),
        RateWidget(theme=Theme.default()),
        ElapsedTimeWidget(theme=Theme.default()),
        EstimatedTimeWidget(theme=Theme.default()),
    ]

    bar = Bar(total=200)
    view = View(bar, widgets=widgets)
    Progress.add_bar(bar, view)

    for i in range(1, 200 + 1):
        bar.increment()
        Progress.display()
        time.sleep(0.01)

    Progress.close()


def example_11():
    print("=== Example 11: Progress tree ===")

    for i in progress(
        range(1, 2 + 1), title=lambda item: f"Processing i: {item.value}"
    ):
        for j in progress(
            range(1, 20 + 1),
            title=lambda item: f"Processing j: {item.value}",
            indent=1,
            remove_on_complete=True,
        ):
            for k in progress(
                range(1, 50 + 1),
                title=lambda item: f"Processing k: {item.value}",
                indent=2,
                remove_on_complete=True,
            ):
                time.sleep(0.001)

    Progress.instance().close()


def example_12():
    print("=== Example 12: Spinner Styles ===")

    with Progress:
        bars = []
        for spinner_style, use_unicode in [
            ("snake", True),
            ("dots", True),
            ("arrows", True),
            ("bouncing", True),
            ("spinner", False),
        ]:
            widgets = [
                TitleWidget(f"Loading ({spinner_style})", theme=Theme.default()),
                SpinnerWidget(
                    style=spinner_style, use_unicode=use_unicode, theme=Theme.default()
                ),
                CounterWidget(theme=Theme.default()),
                ElapsedTimeWidget(theme=Theme.default()),
            ]

            bar = Progress.create_bar(total=0, widgets=widgets)
            bars.append(bar)

        widgets = [
            TitleWidget(f"Loading (custom frames)", theme=Theme.default()),
            SpinnerWidget(frames=["🌍", "🌎", "🌏"], theme=Theme.default()),
            CounterWidget(theme=Theme.default()),
            ElapsedTimeWidget(theme=Theme.default()),
        ]

        bar = Progress.create_bar(total=0, widgets=widgets)
        bars.append(bar)

        for i in range(1, 30 + 1):
            for bar in bars:
                bar.increment()
            time.sleep(0.1)
            Progress.display()


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)

    for i in range(0, 12 + 1):
        if i != 0:
            time.sleep(1)
        globals()[f"example_{i}"]()
