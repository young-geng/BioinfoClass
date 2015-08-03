from __future__ import division
import numpy as np
import multiprocessing
from sys import stdin, stdout, version_info

# For compatibility with python3.
if version_info[0] == 3:
  from functools import reduce
  xrange = range
  __map = map
  map = lambda *args: list(__map(*args))
  __filter = filter
  filter = lambda *args: list(__filter(*args))


SYMBOLS = ['A', 'C', 'G', 'T']
SYMBOL_PAIRS = {
    'A': 'T',
    'T': 'A',
    'C': 'G',
    'G': 'C'
}

SYMBOL_INDEX = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3
}


class FunctionWrapper(object):
  """Wrap a function for use with parallelized map.
  """

  def __init__(self, func, **kwargs):
    """Construct the function oject.

    Args:
      func: a top level function, or a picklable callable object.
      **kwargs: Any additional required enviroment data.
    """
    self.func = func
    self.kwargs = kwargs

  def __call__(self, obj):
    return self.func(obj, **self.kwargs)


def ParallelMap(func, iterable_object, num_processes=1):
  """Parallelized map function based on python process

  Args:
    func: Pickleable callable object that takes one parameter.
    iterable_object: An iterable of elements to map the function on.
    num_processes: Number of process to use. When num_processes is 1,
                   no new processes will be created.
  Returns:
    The list resulted in calling the func on all objects in the original list.
  """
  if num_processes == 1:
    return map(func, iterable_object)
  process_pool = multiprocessing.Pool(num_processes)
  results = process_pool.map(func, iterable_object)
  process_pool.close()
  process_pool.join()
  return results


def SlideSubstrings(text, k):
  """Iterate all substring of length k in text in order.

  Args:
    text: a string
    k: length of substring in integer.
  Returns:
    A generator object that iterates through all substring in order.
  """
  for i in xrange(len(text) - k + 1):
    yield text[i:i + k]


def ReverseComp(text):
  """Reverse complement of DNA string text.
  """
  ans = ''
  for c in text:
    ans = SYMBOL_PAIRS[c] + ans
  return ans

def HammingDistance(text1, text2):
  """Hamming distance of string text1 and text2.
  """
  assert len(text1) == len(text2)
  distance = 0
  for i in xrange(len(text1)):
    if text1[i] != text2[i]:
      distance += 1
  return distance


def FindPatternWithMismatch(text, pattern, num_mismatch=0):
  """Find a pattern in text with at most num_mismatch mismatched characters.

  Args:
    text: a string.
    pattern: a string.
    num_mismatch: maximum number of mismached characters allowed.
  Returns:
    A list of indices where the pattern is found.
  """
  positions = []
  for i in xrange(len(text) - len(pattern) + 1):
    if HammingDistance(pattern, text[i:i + len(pattern)]) <= num_mismatch:
      positions.append(i)
  return positions


def FindClump(text, k, window_size, num_occurrences):
  """Find all the patterns where num_occurrences of pattern exist
     in a window of size window_size.

  Args:
    text: a DNA string.
    k: length of kmer pattern.
    window_size: size of clump window.
    num_occurrences: minimum number of occurrences for pattern in a windows.
  Returns:
    A list of patterns.
  """
  count_table = {}
  clumps = {}
  for pattern in SlideSubstrings(text[:window_size], k):
    if pattern in count_table:
      count_table[pattern] += 1
    else:
      count_table[pattern] = 1
  for pattern in count_table:
    if count_table[pattern] >= num_occurrences:
      clumps[pattern] = None

  for i in xrange(window_size - k + 1, len(text) - k + 1):
    original_i = i - (window_size - k + 1)
    count_table[text[original_i:original_i + k]] -= 1
    if text[i:i + k] in count_table:
      count_table[text[i:i + k]] += 1
    else:
      count_table[text[i:i + k]] = 1
    if count_table[text[i:i + k]] >= num_occurrences:
      clumps[text[i:i + k]] = None

  return sorted(list(clumps.keys()))


def FrequencyArray(text, k):
  """Generate a count array for all possible kmers.
  """
  patterns = sorted(GenerateNeighbors('A' * k, k))
  count = {}
  for pattern in SlideSubstrings(text, k):
    if pattern in count:
      count[pattern] += 1
    else:
      count[pattern] = 1

  count_array = []
  for pattern in patterns:
    if pattern in count:
      count_array.append(count[pattern])
    else:
      count_array.append(0)
  return count_array

def PatternToNumber(pattern):
  total = 0
  for c in pattern:
    total = total * 4 + SYMBOL_INDEX[c]
  return total


def NumberToPattern(number, k):
  pattern = ''
  while number > 0:
    pattern = SYMBOLS[number % 4] + pattern
    number = number // 4
  return 'A' * (k - len(pattern)) + pattern


def GenerateNeighbors(text, dist):
  """Generate all neighbors of a dna string text within Hamming distance dist.

  Args:
    text: a dna string.
    dist: maximum Hamming distance.
  Returns:
    A list of dna strings.
  """
  if len(text) == 0:
    return ['']
  if dist == 0:
    return [text]

  neighbors = []
  for symbol in SYMBOLS:
    neighbors += [symbol + substr for substr in GenerateNeighbors(text[1:], dist - int(symbol != text[0]))]
  return neighbors


def MostFrequentKMerWithMismatch(text, k, num_mismatch=0):
  """Find most frequent kmer in text with mismatch.

  Args:
    text: a DNA string.
    k: length of kmer.
    num_mismatch: maximum number of mismatch allowed.
  Returns:
    A list of all most frequent kmers.
  """
  kmer_count = {}
  for i in xrange(len(text) - k + 1):
    for kmer in GenerateNeighbors(text[i:i+k], num_mismatch):
      if kmer in kmer_count:
        kmer_count[kmer] += 1
      else:
        kmer_count[kmer] = 1

  result = []
  max_count = 0
  for kmer in kmer_count:
    count = kmer_count[kmer]
    if count > max_count:
      max_count = count
      result = [kmer]
    elif count == max_count:
      result.append(kmer)
  return result


def MostFrequentKMerWithMismatchAndReverseComp(text, k, num_mismatch=0):
  """Find most frequent kmer in text with mismatch and reverse complement.

  Args:
    text: a DNA string.
    k: length of kmer.
    num_mismatch: maximum number of mismatch allowed.
  Returns:
    A list of all most frequent kmers.
  """
  kmer_count = {}
  for i in xrange(len(text) - k + 1):
    for kmer in GenerateNeighbors(text[i:i+k], num_mismatch):
      if kmer in kmer_count:
        kmer_count[kmer] += 1
      else:
        kmer_count[kmer] = 1

  revcomp = ReverseComp(text)
  for i in xrange(len(revcomp) - k + 1):
    for kmer in GenerateNeighbors(revcomp[i:i+k], num_mismatch):
      if kmer in kmer_count:
        kmer_count[kmer] += 1
      else:
        kmer_count[kmer] = 1


  result = []
  max_count = 0
  for kmer in kmer_count:
    count = kmer_count[kmer]
    if count > max_count:
      max_count = count
      result = [kmer]
    elif count == max_count:
      result.append(kmer)
  return result

def KDMotif(text_list, k, num_mismatch=0):
  """Compute all motif occurs in all strings in text_list with at
     most num_mismatch mismatches.

  Args:
    text_list: a list of DNA strings.
    k: length of motif.
    num_mismatch: maximum number of mismatches allowed.
  Returns:
    A list of motif strings.
  """
  count_table = {}
  for text_idx in xrange(len(text_list)):
    text = text_list[text_idx]
    for i in xrange(len(text) - k + 1):
      for kmer in GenerateNeighbors(text[i:i + k], num_mismatch):
        if kmer in count_table:
          count_table[kmer][text_idx] = 1
        else:
          count_table[kmer] = [0] * len(text_list)
          count_table[kmer][text_idx] = 1
  
  return sorted([kmer for kmer in count_table if sum(count_table[kmer]) == len(text_list)])



def ComputeMatrixEntropy(kmer_matrix):
  """Compute the entropy of a kmer matrix.

  Args:
    kmer_matrix: a numpy array of DNA characters. Each row is a kmer.
  Returns:
    The entropy of kmer_matrix.
  """
  kmer_matrix = np.array(map(list, kmer_matrix), dtype=str)
  entropy = 0
  for symbol in SYMBOLS:
    p = np.sum(kmer_matrix == symbol, axis=0) / kmer_matrix.shape[0]
    for i in p:
      if i != 0:
        entropy += -i * np.log(i) / np.log(2)
  return entropy


def BruteForceMedianStringWorker(pattern, text_list):
  """Worker for brute force media string.
  """
  total_dist = 0
  for text in text_list:
    min_dist = len(pattern)
    for i in xrange(len(text) - len(pattern) + 1):
      if HammingDistance(pattern, text[i:i + len(pattern)]) < min_dist:
        min_dist = HammingDistance(pattern, text[i:i + len(pattern)])
    total_dist += min_dist
  return total_dist


def BruteForceMedianString(text_list, k):
  """Brute force to find best motifs by iterating through all possible patterns.

  Args:
    text_list: a list of DNA strings.
    k: length of motif.
  Returns:
    A list of motif strings.
  """
  patterns = GenerateNeighbors('A' * k, k)
  mapper = FunctionWrapper(
      BruteForceMedianStringWorker,
      text_list=text_list
  )
  all_dist = zip(ParallelMap(mapper, patterns), patterns)

  min_dist = min(all_dist, key=lambda x: x[0])[0]
  return [x[1] for x in all_dist if x[0] == min_dist]



def PatternProbability(pattern, probability_matrix):
  """Compute the probability of pattern given a probability matrix.

  Args:
    pattern: a DNA string.
    probability_matrix: a numpy matrix with probabilities of each position stored
                        in columns.
  Returns:
    The probability of pattern in a float.
  """
  p = 1
  for i in xrange(len(pattern)):
    p *= probability_matrix[SYMBOL_INDEX[pattern[i]], i]

  return p



def FindMostProbablePattern(text, probability_matrix, k):
  """Find the most probable pattern in text given a profile probability matrix.

  Args:
    test: a DNA string.
    probability_matrix: a numpy matrix with probabilities of each position stored
                        in columns.
    k: length of pattern.
  Returns:
    The most probable DNA pattern as a string.
  """
  best_p = 0
  best_pattern = None
  best_sum = 0
  for pattern in SlideSubstrings(text, k):
    p = PatternProbability(pattern, probability_matrix)
    if p == 0 and best_pattern is None:
      p_sum = 1
      for i in xrange(len(pattern)):
        p_sum += probability_matrix[SYMBOL_INDEX[pattern[i]], i]
      if p_sum > best_sum:
        best_sum = p_sum
        best_pattern = pattern
    elif p > best_p:
      best_p = p
      best_pattern = pattern
  return best_pattern

def GenerateProbabilityMatrix(text_list):
  """Generate an adjusted probability distribution matrix for 4 nucleotides.

  Args:
    text_list: A list of dna strings with same length.
  Returns:
    A numpy 4 * n array where each column stores a probability distribution
    in one position.
  """
  probability = []
  text_matrix = np.array(map(list, text_list), dtype=str)
  for symbol in SYMBOLS:
    probability.append(np.sum(text_matrix == symbol, axis=0) / len(text_list))

  return np.vstack(probability)

def GenerateAdjustedProbabilityMatrix(text_list):
  """Generate an adjusted probability distribution matrix for 4 nucleotides
     with adjustment of adding 1 count.

  Args:
    text_list: A list of dna strings with same length.
  Returns:
    A numpy 4 * n array where each column stores a probability distribution
    in one position.
  """
  probability = []
  text_matrix = np.array(map(list, text_list), dtype=str)
  for symbol in SYMBOLS:
    probability.append((np.sum(text_matrix == symbol, axis=0) + 1) / (text_matrix.shape[0] + 1))
  return np.vstack(probability)

def ScoreMotifs(motifs):
  """Compute the score of motifs.

  Args:
    motifs: A list of strings.
  Returns:
    An integer score.
  """
  motif_matrix = np.array(map(list, motifs), dtype=str)
  symbol_count = []
  for symbol in SYMBOLS:
    symbol_count.append(np.sum(motif_matrix == symbol, axis=0))
  return np.sum(len(motifs) - np.max(np.vstack(symbol_count), axis=0))


def GreedyMotifSearch(text_list, k, t, adjust_probability=False):
  """Greedy search the motif of length k in first t elements
     of dna string text_list.

  Args:
    text_list: a list of dna string.
    k: length of motif pattern.
    t: Number of dna strings to search in.
    adjust_probability: A boolean switch for adjusting probability matrix.
  Returns:
    A list of strings as the motif.
  """
  assert 1 <= t <= len(text_list)

  probability_matrix_func = GenerateProbabilityMatrix
  if adjust_probability:
    probability_matrix_func = GenerateAdjustedProbabilityMatrix

  best_motifs = None
  best_score = k * t
  for first_pattern in SlideSubstrings(text_list[0], k):
    motifs = [first_pattern]
    for i in xrange(1, t):
      probability_matrix = probability_matrix_func(motifs)
      motifs.append(FindMostProbablePattern(text_list[i], probability_matrix, k))
    if ScoreMotifs(motifs) < best_score:
      best_score = ScoreMotifs(motifs)
      best_motifs = motifs
  return best_motifs


def CoordinateDescentMotifSearchWorker(initial_motifs, text_list, k, t):
  """Perform one coordinate descent algoritm for iteratively improve motif.
  
  Args:
    initial_motifs: a list of strings as initial motifs.
    text_list: a list of dna strings.
    k: length of motif.
    t: number of dna strings to search in.
  Returns:
    A list of dna strings motif.
  """
  text_list = text_list[:t]

  motifs = initial_motifs
  while True:
    last_motifs = motifs
    probability_matrix = GenerateAdjustedProbabilityMatrix(motifs)
    motifs = [
        FindMostProbablePattern(text, probability_matrix, k) for text in text_list
    ]
    if reduce(lambda x, y: x and y, map(lambda x: x[0] == x[1], zip(last_motifs, motifs))):
      return motifs


def CoordinateDescentMotifSearch(text_list, k, t, num_restarts):
  """Use coordinate descent algoritm for iteratively improve motif.
  
  Args:
    text_list: a list of dna strings.
    k: length of motif.
    t: number of dna strings to search in.
    num_restarts: number of restarts.
  Returns:
    A list of dna strings motif.
  """
  mapper = FunctionWrapper(
      CoordinateDescentMotifSearchWorker,
      text_list=text_list,
      k=k,
      t=t
  )
  motifs_list = []
  for _ in xrange(num_restarts):
    motifs = []
    for text in text_list[:t]:
      start = np.random.randint(len(text) - k + 1)
      motifs.append(text[start:start + k])
    motifs_list.append(motifs)

  results = ParallelMap(mapper, motifs_list)
  best_motifs = None
  best_score = k * t + 1
  for motifs in results:
    if ScoreMotifs(motifs) <= best_score:
      best_motifs = motifs
      best_score = ScoreMotifs(motifs)
  return best_motifs

def GibbsSamplerMotifSearch(text_list, k, t, num_iterations, num_restarts):
  """Randomized descent with Gibbs sampler.

  Args:
    text_list: a list of dna strings.
    k: length of motif.
    t: number of dnas to search.
    num_iterations: number of iterations to run the improvement.
    num_restarts: number of restarts to run the algoritm.
  Returns:
    A list of strings as motifs.
  """
  text_list = text_list[:t]
  best_motifs = None
  best_score = k * t
  for _ in xrange(num_restarts):
    motifs = []
    for text in text_list:
      start = np.random.randint(len(text) - k + 1)
      motifs.append(text[start:start + k])
    for iteration in xrange(num_iterations):
      dna_index = np.random.randint(t)
      probability_matrix = GenerateAdjustedProbabilityMatrix(motifs)
      probability_dist = []
      patterns = []
      for pattern in SlideSubstrings(text_list[dna_index], k):
        probability_dist.append(PatternProbability(pattern, probability_matrix))
        patterns.append(pattern)
      probability_dist = np.array(probability_dist)
      probability_dist = probability_dist / np.sum(probability_dist)
      motifs[dna_index] = patterns[np.random.choice(len(patterns), p=probability_dist)]

      if ScoreMotifs(motifs) < best_score:
        best_score = ScoreMotifs(motifs)
        best_motifs = motifs[:]
  return best_motifs


def ReadLineStripped():
  line = ''
  while len(line) == 0:
    line = stdin.readline().strip()
  return line

def ReadIntegers():
  return map(int, ReadLineStripped().split())

def ReadAllLines():
  lines = []
  for line in stdin:
    if len(line.strip()) == 0:
      break
    lines.append(line.strip())
  return lines

def PrintList(l, delimiter=' '):
  for elem in l:
    stdout.write(('{}' + delimiter).format(elem))
  if delimiter != '\n':
    stdout.write('\n')

def Print(elem):
  stdout.write('{}\n'.format(elem))


