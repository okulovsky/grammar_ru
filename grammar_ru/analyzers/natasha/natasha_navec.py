import os
from navec import Navec


class NatashaNavec:
    navec = None

    def get_navec():
        if not NatashaNavec.navec:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'models/navec_news_v1_1B_250K_300d_100q.tar')
            NatashaNavec.navec = Navec.load(path)

        return NatashaNavec.navec
