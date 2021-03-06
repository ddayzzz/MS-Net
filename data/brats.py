import os
import random
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def _label_decomp(label_vol, num_cls):
    """
    decompose label for softmax classifier
    original labels are batchsize * W * H * 1, with label values 0,1,2,3...
    this function decompse it to one hot, e.g.: 0,0,0,1,0,0 in channel dimension
    numpy version of tf.one_hot
    """
    one_hot = []
    for i in range(num_cls):
        _vol = np.zeros(label_vol.shape)
        _vol[label_vol == i] = 1
        one_hot.append(_vol)

    if label_vol.ndim == 4:
        return np.concatenate(one_hot, axis=-1)
    elif label_vol.ndim == 3:
        return np.stack(one_hot, axis=-1)

class BrainSegmentationDataset(object):
    """Brain MRI dataset for FLAIR abnormality segmentation"""

    in_channels = ['t1', 't1ce', 't2', 'flair']
    out_channels = 1

    def __init__(
            self,
            images_dir,
            batch_size,
            subset="train",
            validation_cases=10,
            seed=42,
            only_whole_tumor_label=False
    ):
        """

        :param images_dir:
        :parm batch_size:
        :param subset:
        :param validation_cases:
        :param seed:
        """
        assert subset in ["all", "train", "validation"]
        print("reading {} images...".format(subset))

        subject_dirs = glob.glob(f'{images_dir}/*')
        subject_names = [os.path.basename(n) for n in subject_dirs]

        self.patients = subject_names

        # select cases to subset
        if not subset == "all":
            random.seed(seed)  # 确保每次实验数据是一致的
            validation_patients = random.sample(self.patients, k=validation_cases)
            if subset == "validation":
                self.patients = validation_patients
            else:
                self.patients = sorted(
                    list(set(self.patients).difference(validation_patients))
                )
        print(f'Num of patients: {len(self.patients)}')

        if only_whole_tumor_label:
            print("Only using WT label")
        self.volumes = []
        for subject_name in self.patients:
            img = f"{images_dir}/{subject_name}/{subject_name}.npy"
            mask = f"{images_dir}/{subject_name}/{subject_name}_seg.npy"
            if only_whole_tumor_label:
                # 第0标签是 WT，但是需要保持对应的维度
                mask = np.load(mask)[1:-1, :, :, 0:1]
                onehot_mask = _label_decomp(mask, num_cls=2)
                # 这里需要变成 one-hot 的形式
                self.volumes.append((np.load(img)[1:-1, :, :, :], onehot_mask))
            else:
                self.volumes.append((np.load(img)[1:-1, :, :, :], np.load(mask)[1:-1, :, :, :]))
        # 对应样本 slice 的被选中的概率与slice对应的 mask 的概率有关
        self.slice_weights = [m.sum(axis=-1).sum(axis=-1).sum(axis=-1) for v, m in self.volumes]
        self.slice_weights = [
            (s + (s.sum() * 0.1 / len(s))) / (s.sum() * 1.1) for s in self.slice_weights
        ]

        # add channel dimension to masks
        # self.volumes = [(v, m[..., np.newaxis]) for (v, m) in self.volumes]

        print("done creating {} dataset: {}".format(subset, ', '.join(self.patients)))

        # create global index for patient and slice (idx -> (p_idx, s_idx))
        # 将 slice 扩大，这样就有更多的数据量，这里还需要确定patch的数量是多少，相当于分块的作用, 这里不分块了吧，相当于一个patch有一个slice
        num_slices = [v.shape[0] for v, m in self.volumes]
        # patient_slice_index : [(pat_ind, slice_id), ...] slice_id \in [0, num_slice - 1]
        self.patient_slice_index = list(
            zip(
                sum([[i] * num_slices[i] for i in range(len(num_slices))], []),
                sum([list(range(x)) for x in num_slices], []),
            )
        )

        # slices, masks = [], []
        # for i in range(len(self.patient_slice_index)):
        #     patient = self.patient_slice_index[i][0]
        #     slice_n = self.patient_slice_index[i][1]
        #
        #     v, m = self.volumes[patient]
        #     slices.append(v[slice_n])
        #     masks.append(m[slice_n])
        # slices = np.stack(slices)
        # masks = np.stack(masks)
        # 创建 tensorflow 格式的数据集
        # with tf.device("/cpu:0"):
        #     dataset = tf.data.Dataset.from_tensor_slices({'image': slices, 'mask': masks}) \
        #         .shuffle(seed=seed, buffer_size=128) \
        #         .batch(batch_size) \
        #         .repeat()  # 一直循环下去
        # self.tf_dataset = dataset

    def get_slice(self, idx):
        patient = self.patient_slice_index[idx][0]
        slice_n = self.patient_slice_index[idx][1]

        v, m = self.volumes[patient]
        image = v[slice_n]
        mask = m[slice_n]
        return image, mask

    def get_one_batch(self, batch_size):
        images, masks = [], []
        chosen = np.random.choice(len(self.patient_slice_index), size=batch_size, replace=False)
        for i in chosen:
            img, mask = self.get_slice(i)
            images.append(img)
            masks.append(mask)
        return np.stack(images), np.stack(masks)


class BrainSegmentationKFoldDataset(object):
    """Brain MRI dataset for FLAIR abnormality segmentation"""

    in_channels = ['t1', 't1ce', 't2', 'flair']
    out_channels = 1

    def __init__(
            self,
            images_dir,
            batch_size,
            num_fold,
            fold_id,
            subset="train",
            seed=42,
            only_whole_tumor_label=False
    ):
        """

        :param images_dir:
        :parm batch_size:
        :param subset:
        :param validation_cases:
        :param seed:
        """
        assert subset in ["train", "validation"]
        print("reading {} images...".format(subset))

        subject_dirs = glob.glob(f'{images_dir}/*')
        subject_names = [os.path.basename(n) for n in subject_dirs]

        self.patients = subject_names

        # select cases to subset
        # if not subset == "all":
        #     random.seed(seed)  # 确保每次实验数据是一致的
        #     validation_patients = random.sample(self.patients, k=validation_cases)
        #     if subset == "validation":
        #         self.patients = validation_patients
        #     else:
        #         self.patients = sorted(
        #             list(set(self.patients).difference(validation_patients))
        #         )
        print(f'Num of patients: {len(self.patients)}')

        # 生成 kfold 的数据集，按照等分的原则

        print("Generating KFold list...")
        kf = KFold(n_splits=num_fold, shuffle=True, random_state=seed)
        kf_lists = list(kf.split(np.arange(len(self.patients))))
        for i, (train_ids, validation_ids) in enumerate(kf_lists):
            print(f"\tFold {i}:")
            print(f'\t\tTrain ids: {", ".join([self.patients[sid] for sid in train_ids])}')
            print(f'\t\tValidation ids: {", ".join([self.patients[sid] for sid in validation_ids])}')
        # 接着按照对应的 subset 进行筛选
        fold_train_val = kf_lists[fold_id]
        if subset == 'train':
            self.patients = [self.patients[sid] for sid in fold_train_val[0]]
        else:
            self.patients = [self.patients[sid] for sid in fold_train_val[1]]
        print(f"Selecting Fold: {fold_id}")

        if only_whole_tumor_label:
            print("Only using WT label")
        self.volumes = []
        for subject_name in self.patients:
            img = f"{images_dir}/{subject_name}/{subject_name}.npy"
            mask = f"{images_dir}/{subject_name}/{subject_name}_seg.npy"
            if only_whole_tumor_label:
                mask = np.load(mask)[1:-1, :, :, 0:1]
                onehot_mask = _label_decomp(mask, num_cls=2)
                # 这里需要变成 one-hot 的形式
                self.volumes.append((np.load(img)[1:-1, :, :, :], onehot_mask))
            else:
                self.volumes.append((np.load(img)[1:-1, :, :, :], np.load(mask)[1:-1, :, :, :]))
        # 对应样本 slice 的被选中的概率与slice对应的 mask 的概率有关
        self.slice_weights = [m.sum(axis=-1).sum(axis=-1).sum(axis=-1) for v, m in self.volumes]
        self.slice_weights = [
            (s + (s.sum() * 0.1 / len(s))) / (s.sum() * 1.1) for s in self.slice_weights
        ]

        # add channel dimension to masks
        # self.volumes = [(v, m[..., np.newaxis]) for (v, m) in self.volumes]

        print("done creating {} dataset: {}".format(subset, ', '.join(self.patients)))

        # create global index for patient and slice (idx -> (p_idx, s_idx))
        # 将 slice 扩大，这样就有更多的数据量，这里还需要确定patch的数量是多少，相当于分块的作用, 这里不分块了吧，相当于一个patch有一个slice
        num_slices = [v.shape[0] for v, m in self.volumes]
        # patient_slice_index : [(pat_ind, slice_id), ...] slice_id \in [0, num_slice - 1]
        self.patient_slice_index = list(
            zip(
                sum([[i] * num_slices[i] for i in range(len(num_slices))], []),
                sum([list(range(x)) for x in num_slices], []),
            )
        )

        # slices, masks = [], []
        # for i in range(len(self.patient_slice_index)):
        #     patient = self.patient_slice_index[i][0]
        #     slice_n = self.patient_slice_index[i][1]
        #
        #     v, m = self.volumes[patient]
        #     slices.append(v[slice_n])
        #     masks.append(m[slice_n])
        # slices = np.stack(slices)
        # masks = np.stack(masks)
        # 创建 tensorflow 格式的数据集
        # with tf.device("/cpu:0"):
        #     dataset = tf.data.Dataset.from_tensor_slices({'image': slices, 'mask': masks}) \
        #         .shuffle(seed=seed, buffer_size=128) \
        #         .batch(batch_size) \
        #         .repeat()  # 一直循环下去
        # self.tf_dataset = dataset

    def get_slice(self, idx):
        patient = self.patient_slice_index[idx][0]
        slice_n = self.patient_slice_index[idx][1]

        v, m = self.volumes[patient]
        image = v[slice_n]
        mask = m[slice_n]
        return image, mask

    def get_one_batch(self, batch_size):
        images, masks = [], []
        chosen = np.random.choice(len(self.patient_slice_index), size=batch_size, replace=False)
        for i in chosen:
            img, mask = self.get_slice(i)
            images.append(img)
            masks.append(mask)
        return np.stack(images), np.stack(masks)


if __name__ == '__main__':
    dataset = BrainSegmentationDataset(
        images_dir=r'/home/wangshu/datasets/brats/splited_by_ManufacturerModelName_preprocessed/train/signa_excite_1_5',
        validation_cases=1,
        batch_size=1,
        only_whole_tumor_label=True)

    num_slices = np.bincount([p[0] for p in dataset.patient_slice_index])
    f = dataset.get_one_batch(32)
    for i in range(1, len(dataset.patient_slice_index), 100):
        image, mask = dataset.get_slice(i)

        plt.subplot(421)
        plt.imshow(image[:, :, 0], cmap='gray')
        plt.subplot(422)
        plt.imshow((mask[:, :, 0] * 255).astype(np.uint8), cmap='gray')
        plt.subplot(423)
        plt.imshow(image[:, :, 1], cmap='gray')
        plt.subplot(424)
        plt.imshow((mask[:, :, 0] * 255).astype(np.uint8), cmap='gray')
        plt.subplot(425)
        plt.imshow(image[:, :, 2], cmap='gray')
        plt.subplot(426)
        plt.imshow((mask[:, :, 1] * 255).astype(np.uint8), cmap='gray')

        plt.subplot(427)
        plt.imshow(image[:, :, 3], cmap='gray')
        plt.subplot(428)
        plt.imshow((mask[:, :, 2] * 255).astype(np.uint8), cmap='gray')

        plt.show()
        # plt.savefig(f'/tmp/1.png')
