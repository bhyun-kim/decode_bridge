from copy import deepcopy
import numpy as np
import torch

class RandomCrop1D(object):
    def __init__(self, output_size):
        """
        Args:
            output_size (int): Desired output size.
        """
        self.output_size = output_size

    def __call__(self, input):
        """
        Args:
            input (dict): 'measurement' and 'label' keys

        Returns:
            output (dict): 'measurement' and 'label' keys
        """
        output = deepcopy(input)

        # get measurement length
        measurement = input['measurement']
        measure_len = measurement.shape[0]

        start = np.random.randint(0, measure_len - self.output_size)

        # get start index
        if measure_len > self.output_size:
            end = start + self.output_size
            output['measurement'] = measurement[start:end]
        else:
            # pad data
            end = start + measure_len
            output['measurement'] = np.zeros((self.output_size))
            output['measurement'][start:end] = measurement

        return output

# PyTorch transforms class adding noise to data
class RandomNoise(object):
    def __init__(self, max_noise_level=0.01):
        """
        Args:
            max_noise_level (float): maximum noise level to add to data
        """
        self.max_noise_level = max_noise_level

    def __call__(self, input):
        """
        Args:
            input (dict): 'measurement' and 'label' keys

        Returns:
            output (dict): 'measurement' and 'label' keys
        """

        output = deepcopy(input)

        # get data length
        measurement = input['measurement']
        measure_len = measurement.shape[0]

        # generate noise
        noise_level = np.random.uniform(0, self.max_noise_level)
        noise = np.random.uniform(0, np.max(measurement), measure_len) * noise_level
        output['measurement'] = measurement + noise

        return output

# PyTorch transforms class randomly dropping data
class RandomDrop(object):
    def __init__(self,
                 prop=0.2,
                 min_length=64,
                 max_length=256):
        """
        Args:
            prop (float): proportion of data to drop [0, 1]
            min_length (float): minimum length to drop from data
            max_length (float): maximum length to drop from data
        """
        self.prop = prop
        self.min_length = min_length
        self.max_length = max_length

    def __call__(self, input):
        """
        Args:
            input (dict): 'measurement' and 'label' keys

        Returns:
            output (dict): 'measurement' and 'label' keys
        """

        output = deepcopy(input)

        # determin if drop data or not
        if np.random.uniform(0, 1) > self.prop:
            return output

        # get data length
        measurement = input['measurement']
        measure_len = measurement.shape[0]

        drop_from = np.random.randint(0, measure_len - self.max_length)
        drop_length = np.random.randint(self.min_length, self.max_length)

        # drop data
        output['measurement'][drop_from:drop_from+drop_length] = 0

        return output

class ToTensor(object):
    def __call__(self, input):
        """
        Args:
            input (dict): 'measurement' and 'label' keys

        Returns:
            output (dict): 'measurement' and 'label' keys
        """


        return {'measurement': torch.from_numpy(input['measurement']).float(),
                'label': torch.tensor(input['label']).float()}