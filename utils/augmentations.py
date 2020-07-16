import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
import torchvision.transforms.functional as FT


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size,
                                 self.size))
        return image, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels

class RandomFlip(object):
    def __call__(self, image, boxes, classes):
        if random.randint(2):
            new_image = FT.hflip(image)
            
            ## Flip boxes
            new_boxes = boxes
            new_boxes[:, 0] = image.width - boxes[:, 0] - 1
            new_boxes[:, 2] = image.width - boxes[:, 2] - 1
            new_boxes = new_boxes[:, [2, 1, 0, 3]]

        return new_image, new_boxes

class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)

class CutMix(object):
    def __init__(self, size):
        self.size = float(size)
        
    def rand_bbox(self, size, lamb):
        """ Generate random bounding box 
        Args:
            - size: [width, breadth] of the bounding box
            - lamb: (lambda) cut ratio parameter
        Returns:
            - Bounding box
        """
        W = size[0]
        H = size[1]
        cut_rat = np.sqrt(1. - lamb)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def filter_cropped(self, bboxes, labels, xmin, ymin, xmax, ymax):
        # convert to integer rect x1,y1,x2,y2
        rect = np.array([xmin, ymin, xmax, ymax])

        new_boxes = bboxes.copy() 
        new_boxes *= self.size
        new_labels = labels.copy()

        # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
        inter = intersect(new_boxes, rect)
        overlap = np.zeros(shape=inter.shape, dtype=float)

        area = (new_boxes[:, 2] - new_boxes[:, 0]) * (new_boxes[:, 3] - new_boxes[:, 1])
        overlap = inter / area

        mask = (overlap > 0)

        if mask.any():
            new_boxes = new_boxes[mask, :]
            new_labels = new_labels[mask]
        else:
            mask = ~mask
            return new_boxes[mask, :], new_labels[mask]

        new_boxes[:, 0] = np.clip(new_boxes[:, 0], xmin, xmax)
        new_boxes[:, 2] = np.clip(new_boxes[:, 2], xmin, xmax)
        new_boxes[:, 1] = np.clip(new_boxes[:, 1], ymin, ymax)
        new_boxes[:, 3] = np.clip(new_boxes[:, 3], ymin, ymax)

        mask_x = ((new_boxes[:, 0] == xmin) * (new_boxes[:, 2] == xmin)) + ((new_boxes[:, 0] == xmax) * (new_boxes[:, 2] == xmax))
        mask_y = ((new_boxes[:, 1] == ymin) * (new_boxes[:, 3] == ymin)) + ((new_boxes[:, 1] == ymax) * (new_boxes[:, 3] == ymax))
        mask = mask_x + mask_y

        if mask.any():
            mask = ~mask
            new_boxes = new_boxes[mask, :]
            new_labels = new_labels[mask]

        new_boxes /= self.size

        return new_boxes, new_labels
        
    def filter_uncropped(self, bboxes, labels, xmin, ymin, xmax, ymax):
        new_boxes = bboxes.copy()
        new_boxes *= self.size
        new_labels = labels.copy()

        # convert to integer rect x1,y1,x2,y2
        rect = np.array([xmin, ymin, xmax, ymax])

        # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
        inter = intersect(new_boxes, rect)
        overlap = np.zeros(shape=inter.shape, dtype=float)

        area = (new_boxes[:, 2] - new_boxes[:, 0]) * (new_boxes[:, 3] - new_boxes[:, 1])
        overlap = inter / area

        mask = (overlap < 1.0)

        if mask.any():
            new_boxes = new_boxes[mask, :]
            new_labels = new_labels[mask]
        else:
            mask = ~mask
            return new_boxes[mask, :], new_labels[mask]

        mask_xmin = ((new_boxes[:, 0] > xmin) * (new_boxes[:, 0] < xmax))
        mask_xmax = ((new_boxes[:, 2] > xmin) * (new_boxes[:, 2] < xmax))
        mask_ymin = ((new_boxes[:, 1] > ymin) * (new_boxes[:, 1] < ymax))
        mask_ymax = ((new_boxes[:, 3] > ymin) * (new_boxes[:, 3] < ymax))

        mask = mask_xmin * mask_ymin * mask_ymax

        if mask.any():
            new_boxes[mask, 0] = xmax

        mask = mask_xmax * mask_ymin * mask_ymax

        if mask.any():
            new_boxes[mask, 2] = xmin

        mask = mask_ymin * mask_xmin * mask_xmax

        if mask.any():
            new_boxes[mask, 1] = ymax

        mask = mask_ymax * mask_xmin * mask_xmax

        if mask.any():
            new_boxes[mask, 3] = ymin

        new_boxes /= self.size

        return new_boxes, new_labels

    def generate_cutmix_image(self, image_batch, image_batch_boxes, image_batch_labels, beta=1.0):
        """ Generate a CutMix augmented image from a batch 
        Args:
            - image_batch: a batch of input images
            - image_batch_labels: labels corresponding to the image batch
            - beta: a parameter of Beta distribution.
        Returns:
            - CutMix image batch, updated labels
        """
        # generate mixed sample
        lam = np.random.beta(beta, beta)
            
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(image_batch[1].shape, lam)
        image_batch_updated = image_batch.copy()
        
        image_batch_updated[0, bby1:bby2, bbx1:bbx2, :] = image_batch_updated[1, bby1:bby2, bbx1:bbx2, :]
        
        
#         print(bbx1, bby1, bbx2, bby2)
        image_ref_boxes, image_ref_labels = self.filter_uncropped(image_batch_boxes[0], 
                                                                  image_batch_labels[0],
                                                                  bbx1, bby1, bbx2, bby2
                                                                 )
        
        
        image_patch_boxes, image_patch_labels = self.filter_cropped(image_batch_boxes[1],
                                                                    image_batch_labels[1],
                                                                    bbx1, bby1, bbx2, bby2
                                                                   )
        
        image_boxes_updated = np.concatenate([image_ref_boxes, image_patch_boxes])
        image_labels_updated = np.concatenate([image_ref_labels, image_patch_labels])
        
        return image_batch_updated[0], image_boxes_updated, image_labels_updated

    def generate_mixup_image(self, image_batch, image_batch_boxes, image_batch_labels, beta=1.0):
        """ Generate a MixUp augmented image from a batch 
        Args:
            - image_batch: a batch of input images
            - image_batch_labels: labels corresponding to the image batch
            - beta: a parameter of Beta distribution.
        Returns:
            - CutMix image batch, updated labels
        """
        # generate mixed sample
        lam = np.random.beta(beta, beta)
            
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(image_batch[1].shape, lam)
        image_batch_updated = image_batch.copy()
        
        image_batch_updated[0, bby1:bby2, bbx1:bbx2, :] = ( image_batch_updated[0, bby1:bby2, bbx1:bbx2, :] + 
                                                            image_batch_updated[1, bby1:bby2, bbx1:bbx2, :] ) / 2.0
        
        
#         print(bbx1, bby1, bbx2, bby2)
        image_ref_boxes, image_ref_labels = image_batch_boxes[0].copy(), image_batch_labels[0].copy()
        
        
        image_patch_boxes, image_patch_labels = self.filter_cropped(image_batch_boxes[1],
                                                                    image_batch_labels[1],
                                                                    bbx1, bby1, bbx2, bby2
                                                                   )
        
        image_boxes_updated = np.concatenate([image_ref_boxes, image_patch_boxes])
        image_labels_updated = np.concatenate([image_ref_labels, image_patch_labels])
        
        return image_batch_updated[0], image_boxes_updated, image_labels_updated
        
    def __call__(self, ref_image, ref_boxes, ref_labels, rand_image, rand_boxes, rand_labels):
        batch_img = np.array([ref_image, rand_image])
        batch_boxes = np.array([ref_boxes, rand_boxes])
        batch_labels = np.array([ref_labels, rand_labels])
        
        if random.randint(2):
            return self.generate_cutmix_image(batch_img, batch_boxes, batch_labels, 1.0)
        else:
            return self.generate_mixup_image(batch_img, batch_boxes, batch_labels, 1.0)
        
class SSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
#             RandomFlip(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])
        
        self.augmix = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])
        
        self.cutmix = CutMix(self.size)

    def __call__(self, img, boxes, labels, rand_img=None, rand_boxes=None, rand_labels=None):
        if rand_img is None:
            return self.augment(img, boxes, labels)
        else:
            in_img, in_boxes, in_labels = self.augmix(img, boxes, labels)
            patch_img, patch_boxes, patch_labels = self.augmix(rand_img, rand_boxes, rand_labels)
            
            return self.cutmix(in_img, in_boxes, in_labels, patch_img, patch_boxes, patch_labels)
