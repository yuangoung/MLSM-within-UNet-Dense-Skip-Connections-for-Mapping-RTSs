import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2, dtype=np.float64)

    def Pixel_Accuracy(self):
        OA = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return OA

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) +
            np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix)
        )
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) +
            np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix)
        )
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def Precision_Class(self):
        """
        Per-class precision:
        precision_i = TP_i / (TP_i + FP_i)
                    = diag / column_sum
        """
        precision = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=0)
        return precision

    def Recall_Class(self):
        """
        Per-class recall:
        recall_i = TP_i / (TP_i + FN_i)
                 = diag / row_sum
        """
        recall = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)
        return recall

    def F1_Class(self):
        """
        Per-class F1 score:
        F1_i = 2 * P_i * R_i / (P_i + R_i)
        This is computed from the confusion matrix in a strict class-wise manner.
        """
        precision = self.Precision_Class()
        recall = self.Recall_Class()
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def Mean_F1(self):
        """
        Macro-F1:
        arithmetic mean of per-class F1 scores.
        This is usually stricter than reporting only the foreground F1.
        """
        f1 = self.F1_Class()
        return np.nanmean(f1)

    def Foreground_F1(self, foreground_class=1):
        """
        F1 score for a specific foreground class in segmentation.
        For binary segmentation, foreground_class=1 is commonly used.
        """
        f1 = self.F1_Class()
        return f1[foreground_class]

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype(int) + pre_image[mask].astype(int)
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2, dtype=np.float64)