Dataset Instructions
---

We randomly sample 1000 examples from HotPotQA, MuSiQue, and 2Wiki. In this directory we provide the example indices that we used in our experiments. For better reproducibility, kindly download the respective datasets and sample the examples accordingly. Store the datasets in `datasetname_1000.json` for compatibility with the rest of the repository.

#### 2Wiki data example
```
{
    "_id": "id",
    "type": "type",
    "question": "Question",
    "context": [
      [
        "Title",
        [
          "Paragraph1", "Paragraph2" ...
        ]
      ],
      ...
    ],
    "supporting_facts": [
      [
        "Title",
        Index
      ],
      ...
    ],
    "evidences": [
      [
        "evidence",
        ...
      ],
      ...
    ],
    "answer": "Answer"
  }
```

#### HotpotQA data example
```
{
    "question": "Question",
    "supporting_facts": [
      [
        "Title",
        Index
      ],
      ...
    ],
    "level": "level",
    "context": [
      [
        "Title",
        [
          "Paragraph1",
          ...
        ]
      ],
      ...
    ],
    "answer": "Answer",
    "_id": "id",
    "type": "type"
  },
```

#### MuSiQue data example
```
{
    "id": "id",
    "paragraphs": [
      {
        "idx": idx,
        "title": "Title",
        "paragraph_text": "Paragraph",
        "is_supporting": false
      },
      ...
    ],
    "question": "Question",
    "question_decomposition": [
      {
        "id": id_dec,
        "question": "sub_query",
        "answer": "sub_answer",
        "paragraph_support_idx": id_sup
      },
      ...
    ]
}
```