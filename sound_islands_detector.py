from itertools import groupby, count
from scipy.signal import medfilt

def convert_seconds_to_frames(seconds, frame_size=512, sample_rate=44100.):
    """
    Converts 'seconds' from seconds to frames, according to the duration of the frame
    :param seconds:
    :param frame_size:
    :param sample_rate:
    :return:
    """
    frame_duration = frame_size / sample_rate
    return seconds / frame_duration

def remove_short_sound_islands(sound_islands_candidates, min_note_duration=0.2):
    min_note_duration_frames = convert_seconds_to_frames(min_note_duration)
    sound_islands = [island for island in sound_islands_candidates if
                     island[1] - island[0] > min_note_duration_frames]
    return sound_islands


def remove_short_non_sound_islands(sound_islands_candidates, min_non_sound_frames_island=5,
                                   min_sound_frames_island=20):
    """
    Removes short non-sound islands inside sound islands
    :param sound_islands_candidates:
    :param min_non_sound_frames_island: the minimum length of a non-sound island for being merged with its
        containing sound island
    :param min_sound_frames_island: the minimum length of an island to be considered a part of a sound island,
        useful to avoid merging very short candidate sound islands only because they are close (if they are both
        short they are probably background noise with a pitch, such as a voice)
    :return:
    """
    islands_to_merge_indexes = []
    for iteration_index, sound_island in enumerate(sound_islands_candidates):
        is_not_last_sound_island = iteration_index + 1 < len(sound_islands_candidates)
        if is_not_last_sound_island:
            sound_island_end = sound_island[1]
            sound_island_length = sound_island[1] - sound_island[0]
            next_sound_island = sound_islands_candidates[iteration_index + 1]
            next_sound_island_start = next_sound_island[0]
            next_sound_island_length = next_sound_island[1] - next_sound_island[0]
            not_short_sound_islands = (sound_island_length > min_sound_frames_island) or \
                                      (next_sound_island_length > min_sound_frames_island)
            if next_sound_island_start - sound_island_end < min_non_sound_frames_island and not_short_sound_islands:
                islands_to_merge_indexes.append(iteration_index)
                islands_to_merge_indexes.append(iteration_index + 1)
    # remove duplicates
    islands_to_merge_indexes = list(set(islands_to_merge_indexes))
    new_sound_islands_candidates = [island for index, island in enumerate(sound_islands_candidates)
                                    if index not in islands_to_merge_indexes]
    islands_to_merge_indexes_groups = groupby(islands_to_merge_indexes, key=lambda item, c=count(): item - next(c))
    islands_to_merge_groups = [list(e) for i, e in islands_to_merge_indexes_groups]
    for group in islands_to_merge_groups:
        new_merged_island = [sound_islands_candidates[group[0]][0], sound_islands_candidates[group[-1]][1]]
        new_sound_islands_candidates.append(new_merged_island)
    return new_sound_islands_candidates


class SoundIslandsDetector(object):
    def __init__(self, **kwargs):
        """
        An island is an list with just two elements, in the form of [island_start, island_end] where all the elements
        between island_start and island_end are of the same type, i.e. sound or non-sound.
        Both island_start and island_end are part of the island.
        :param kwargs:
        :return:
        """
        required_keys = ['loudness_envelope', 'pitches', 'pitches_conf']
        for key in required_keys:
            if key not in kwargs.keys():
                raise ValueError('The key %s is required when initializing a SoundIslandsDetector '
                                 'object' % key)
        self.loudness_envelope = kwargs['loudness_envelope']
        self.sound_frames = range(len(self.loudness_envelope))
        self.pitches = kwargs['pitches']
        self.pitches_conf = kwargs['pitches_conf']

    def run(self):
        # initial pruning: detect non sound part of the track
        non_sound_frames = self.detect_non_sound_frames()
        # post processing of the pruning, merge shorts non-sound islands and remove very short notes
        sound_islands = self.post_process_non_sound_frames_detected(non_sound_frames)
        return sound_islands

    def detect_non_sound_frames(self, ratio=0.03, loudness_med_kernel_size=101, pitch_conf_threshold=0.8):
        """
        We define 'non sound frame' an audio frame that is not a note/sound (background noise, voices...).
        We first apply median filtering to the loudness envelope in order to reduce the impact of loudness outliers.
        We then compute max and min of this filtered loudness; these values are used to create a loudness threshold:
        we classify as 'non-sound' all frames whose loudness is below this threshold and whose pitch confidence is
        not particularly high.
        :param ratio: the minimum value of (c_frame_loudness/max_loudness) for c_frame not to be
            considered background noise
        :param loudness_med_kernel_size: the kernel size when computing the median filter (51 is around 500ms)
        :param pitch_conf_threshold: minimum value of pitch confidence to avoid classifying a candidate as non-sound
        :return: the list of 'non-sound' frames
        """
        med_loudness_envelope = medfilt(self.loudness_envelope, loudness_med_kernel_size)
        max_med_loudness_envelope = max(med_loudness_envelope)
        min_med_loudness_envelope = min(med_loudness_envelope)
        threshold_level = min_med_loudness_envelope + (max_med_loudness_envelope - min_med_loudness_envelope) * ratio
        noise_frames = []
        for index, loudness in enumerate(med_loudness_envelope):
            if loudness < threshold_level and self.pitches_conf[index] < pitch_conf_threshold:
                noise_frames.append(index)
        return noise_frames

    def post_process_non_sound_frames_detected(self, non_sound_frames):
        """
        Processes detections of non-sound frames, at first generating 'sound islands' candidates.
        Sound islands candidates are then selected according to their content, length and position.
        It finally returns the sound islands candidates that have successfully passed all the selection phases.
        :param non_sound_frames: list of the indexes of non-sound frames in the audio track
        :return: the final list of sound islands, i.e. segments of the audio track that contain a pitch track
        """
        sound_frames_indexes = [frame_index for frame_index in self.sound_frames
                                if frame_index not in non_sound_frames]
        '''
        Merge into a single list consecutive indexes of 'sound_frames_indexes', so to keep only start_index and
        end_index for each island. Put each of these lists inside a unique list, 'sound_islands_candidates'
        '''
        sound_island_indexes_groups = groupby(sound_frames_indexes, key=lambda item, c=count(): item - next(c))
        sound_frames_groups = [list(e) for i, e in sound_island_indexes_groups]
        sound_islands_candidates = [[x[0], x[-1]] for x in sound_frames_groups]
        # remove detections of short non-sound islands
        sound_islands_candidates = remove_short_non_sound_islands(sound_islands_candidates)
        # remove detections of short sound islands
        sound_islands = remove_short_sound_islands(sound_islands_candidates)
        return sound_islands
