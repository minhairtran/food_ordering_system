import torch.nn as nn
import random
import torch


class SpecAugment(nn.Module):

    def __init__(
        self, freq_masks=0, time_masks=0, freq_width=10, time_width=10, rng=None, mask_value=0.0,
    ):
        super(SpecAugment, self).__init__()

        self._rng = random.Random() if rng is None else rng

        self.freq_masks = freq_masks
        self.time_masks = time_masks

        self.freq_width = freq_width
        self.time_width = time_width

        self.mask_value = mask_value

        if isinstance(time_width, int):
            self.adaptive_temporal_width = False
        else:
            if time_width > 1.0 or time_width < 0.0:
                raise ValueError('If `time_width` is a float value, must be in range [0, 1]')

            self.adaptive_temporal_width = True

    @torch.no_grad()
    def forward(self, x):
        sh = x.shape

        if self.adaptive_temporal_width:
            time_width = max(1, int(sh[2] * self.time_width))
        else:
            time_width = self.time_width

        for idx in range(sh[0]):
            for i in range(self.freq_masks):
                x_left = self._rng.randint(0, sh[1] - self.freq_width)

                w = self._rng.randint(0, self.freq_width)

                x[idx, x_left : x_left + w, :] = self.mask_value

            for i in range(self.time_masks):
                y_left = self._rng.randint(0, sh[2] - time_width)

                w = self._rng.randint(0, time_width)

                x[idx, :, y_left : y_left + w] = self.mask_value

        return x


class SpecCutout(nn.Module):

    def __init__(self, rect_masks=0, rect_time=5, rect_freq=20, rng=None):
        super(SpecCutout, self).__init__()

        self._rng = random.Random() if rng is None else rng

        self.rect_masks = rect_masks
        self.rect_time = rect_time
        self.rect_freq = rect_freq

    @torch.no_grad()
    def forward(self, x):
        sh = x.shape

        for idx in range(sh[0]):
            for i in range(self.rect_masks):
                rect_x = self._rng.randint(0, sh[1] - self.rect_freq)
                rect_y = self._rng.randint(0, sh[2] - self.rect_time)

                w_x = self._rng.randint(0, self.rect_time)
                w_y = self._rng.randint(0, self.rect_freq)

                x[idx, rect_x : rect_x + w_x, rect_y : rect_y + w_y] = 0.0

        return x


class SpectrogramAugmentation(nn.Module):

    def __init__(self,
                 freq_masks=0,
                 time_masks=0,
                 freq_width=10,
                 time_width=10,
                 rect_masks=0,
                 rect_time=5,
                 rect_freq=20,
                 rng=None,
                 mask_value=0.0,):
        super(SpectrogramAugmentation, self).__init__()

        if rect_masks > 0:
            self.spec_cutout = SpecCutout(rect_masks=rect_masks, rect_time=rect_time, rect_freq=rect_freq, rng=rng,)
        else:
            self.spec_cutout = lambda x: x

        if freq_masks + time_masks > 0:
            self.spec_augment = SpecAugment(
                freq_masks=freq_masks,
                time_masks=time_masks,
                freq_width=freq_width,
                time_width=time_width,
                rng=rng,
                mask_value=mask_value,
            )
        else:
            self.spec_augment = lambda x: x


    def forward(self, input_spec):
        augmented_spec = self.spec_cutout(input_spec)
        augmented_spec = self.spec_augment(augmented_spec)
        return augmented_spec
