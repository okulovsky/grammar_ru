from yo_fluq_ds import *
import ipywidgets as widgets


class Task:
    def __init__(self, caption: str, options: List[List[str]] = None, info=None):
        self.caption = caption
        if options is None:
            options = []
        self.options = options
        self.info = info


class TaskResult:
    def __init__(self, task, result):
        self.task = task
        self.result = result


class ITaskProvider:
    def undo(self):
        pass

    def get(self) -> Task:
        raise NotImplementedError()

    def store(self, result: TaskResult):
        raise NotImplementedError()


class DummyTaskProvider(ITaskProvider):
    def __init__(self):
        self.n = 0

    def get(self):
        if self.n > 3:
            return None
        options = []
        for i in range(3):
            rows = []
            options.append(rows)
            for j in range(2):
                rows.append(f'Task {self.n}, row {i}, column {j}')

        task = Task(f'caption {self.n}', options, self.n)
        return task

    def undo(self):
        self.n -= 1

    def store(self, result):
        self.n += 1


class Annotator:
    def __init__(self, provider: ITaskProvider):
        self.provider = provider
        self.html_widget = widgets.HTML()
        self.control_panel = widgets.HBox()
        self.main_panel = widgets.VBox()
        self.current_task = None

    def _align_gui(self, task):
        self.html_widget.value = task.caption

        rows_containers = []
        for row in task.options:
            buttons = []
            for option in row:
                button = widgets.Button(description=option)
                button.on_click(self._vote)
                buttons.append(button)
            container = widgets.HBox(buttons)
            rows_containers.append(container)
        buttons = widgets.VBox(rows_containers)

        self.main_panel.children = (self.html_widget, buttons, self.control_panel)

    def _next_and_align(self):
        self.current_task = self.provider.get()
        if self.current_task is None:
            self._align_gui(Task('Job is done'))
        else:
            self._align_gui(self.current_task)

    def _vote(self, button):
        if self.current_task is None:
            return
        result = button.description
        result = TaskResult(self.current_task, result)
        self.provider.store(result)
        self._next_and_align()

    def _undo(self, _):
        self.provider.undo()
        self._next_and_align()

    def run(self):
        undo_button = widgets.Button(description='UNDO')
        undo_button.on_click(self._undo)
        self.control_panel.children = (undo_button,)
        self._next_and_align()
        return self.main_panel