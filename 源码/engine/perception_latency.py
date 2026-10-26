"""Same full-board recognition; overlap diamonds and bodies only in fast mode."""
from concurrent.futures import ThreadPoolExecutor
import time
from perception_candidate_precise import Perception as BaselinePerception
from perception import Perception as DiamondWorker


class Perception(BaselinePerception):
    def __init__(self):
        super().__init__()
        self.high_speed_mode = False
        self.parallel_diamonds = True
        self._pool = None
        self._diamond_worker = None
        self._diamond_future = None

    def reset(self):
        super().reset()
        self.high_speed_mode = False

    def _scan_diamonds(self, image, origin):
        worker = self._diamond_worker
        worker.item_centers = {}
        worker._item_center_areas = {}
        result = worker.diamonds_full(image, origin=origin)
        return result, dict(worker.item_centers), dict(worker._item_center_areas)

    def diamonds_full(self, image, origin=(0, 0)):
        if self._diamond_future is not None:
            return []  # resolved before this SAME frame is published
        return super().diamonds_full(image, origin=origin)

    def run(self, image, *args, **kwargs):
        started = time.perf_counter()
        parallel = bool(self.high_speed_mode and self.parallel_diamonds)
        if parallel:
            if self._pool is None:
                self._diamond_worker = DiamondWorker()
                self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='same-frame-diamonds')
            box = kwargs.get('restrict_box')
            if box is None:
                crop, origin = image, (0, 0)
            else:
                x0, y0, x1, y1 = box
                crop, origin = image[y0:y1, x0:x1], (x0, y0)
            self._diamond_future = self._pool.submit(self._scan_diamonds, crop, origin)
        try:
            snakes, items, flags = super().run(image, *args, **kwargs)
            if parallel:
                diamonds, centers, areas = self._diamond_future.result()
                items['dia'] = diamonds
                self.item_centers.update(centers)
                self._item_center_areas.update(areas)
                flags['item_centers'].update(centers)
            flags['parallel_diamonds'] = parallel
            flags['ms'] = (time.perf_counter()-started)*1000.
            return snakes, items, flags
        finally:
            # Exceptions must finish the outstanding worker before another
            # frame is allowed to clear its private result dictionaries.
            if self._diamond_future is not None:
                self._diamond_future.result()
            self._diamond_future = None

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
