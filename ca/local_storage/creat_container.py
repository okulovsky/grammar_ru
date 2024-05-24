from pathlib import Path

import container


container1 = container.Container(
        name ='Mnist_container',
        tag ='1',
        entry_point = None,
        dependencies = ['torch', 'torchvision'],
        deployed_folders =['ca'],
        pusher = None,
    )

container1.build()