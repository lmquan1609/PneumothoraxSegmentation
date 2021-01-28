class MaskBinarization:
    def __init__(self):
        self.thresholds = 0.5
    
    def transform(self, preds):
        yield preds > self.thresholds

class SimpleMaskBinarization(MaskBinarization):
    def __init__(self, score_thresholds):
        super().__init__()
        self.thresholds = score_thresholds

    def transform(self, preds):
        for threshold in self.thresholds:
            yield preds > threshold

class DupletMaskBinarization(MaskBinarization):
    def __init__(self, duplets, with_channels=True):
        super().__init__()
        self.thresholds = duplets
        self.dims = (2, 3) if with_channels else (1, 2)
    
    def transform(self, preds):
        for score_threshold, area_threshold in self.thresholds:
            mask = preds > score_threshold
            mask[mask.sum(dim=self.dims) < area_threshold] = 0
            yield mask

class TripletMaskBinarization(MaskBinarization):
    def __init__(self, triplets, with_channels=True):
        super().__init__()
        self.thresholds = triplets
        self.dims = (2, 3) if with_channels else (1, 2)
    
    def transform(self, preds):
        for top_score_threshold, area_threshold, bottom_score_threshold in self.thresholds:
            clf_mask = preds > top_score_threshold
            preds_mask = preds > bottom_score_threshold
            preds_mask[clf_mask.sum(dim=self.dims) < area_threshold] = 0
            yield preds_mask