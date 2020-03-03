# Methods-for-Improvement-of-Automated-Speech-Recognition-Transcripts

Text Alignment and optimal text selection for improvement of automated speech recognition transcripts

### Project description

Automatic Speech Recognition (ASR) has many applications, for instance transcribing interviews and meetings, chatbots, automatic answering for phonecalls etc. The speech-to-text quality has significantly increased in the last years and so has the demand for ASR-based solutions. Tech companies developing ASR systems are in constant pursuit of reaching the best possible transcription quality, and in some experiments even reach human performance.
However, in many cases the output of a particular ASR system contains large amounts of errors. One potential solution would be to go beyond one ASR system and use/combine the output of several systems to produce the transcript. In other contexts (e.g. sentiment analysis), it has been shown that such "ensemble methods" outperform each single participating system. Is the same possible for ASR?

### Goals

- Define a method for comparing ASR system transcriptions on a word level and retrieving the most accurate ones.
- Investigate discrepancies between system transcriptions on a word level.
- Develop a tool for generating an optimized transcription using the outputs from multiple ASR systems.
- Generate optimized transcriptions for a provided set of utterances and calculate the quality  of the meta-system per utterance.

### Data set description

Manual transcriptions with metadata coming from 10 English corpora.
Machine transcriptions produced by 7 ASR systems.
Data is provided in a unified format as a JSON file.

### Data exploration

- explored files received and file structure;
- average wer per corpus and wer distribution per, configuration, machine;
- volume of data per corpusa and configuration;
- are there empty reference texts?;
- are there empty hypothesis texts but not reference texts?;
- do references contain special characters?
- how many references with less than 5 words
- upper cases in reference or hypothesis

### Data cleaning

Cleaning steps:
- lowercase reference and hypothesis
- remove from reference:, ", [, ], {, }
- remove leading, trailing, multiple spaces from hypothesis and reference
- remove sentences with < 4 words
- drop corpus that are a duplicate but with segmentation alterations
- drop reference texts with more than 1 speaker
- keep only columns of interest

### Alignment Methods

In order to select the optimal text alignment of the hypothesis texts was first required. A number of methods were reviewd with the most success coming from:
- Alignment by Entropy
- Alignment by Similarity
- Alignment by Dynamic Programming

### Models

- Alignment by Entropy - see file for walkthrough
- Alignment by Similarity - see file for walkthrough
- Alignment by Dynamic Programming - see file for walkthrough

### Summary
Alignment was achieved.
Optimal Text selection methods could be further investigated.
Optimal Text Selection (based on frequency rate with threshold = 0.35) using only private machines shows some success with a similar word error rate to the best private machines. Further improvements could be made
