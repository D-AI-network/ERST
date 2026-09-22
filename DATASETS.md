# Dataset distribution and citation notes

**Important:** citing a dataset is not the same as having permission to redistribute it. A citation satisfies scholarly attribution; redistribution requires a license or terms that actually permit sharing copies or derivatives.

For a blind/reproducibility repository, the lowest-risk default is:

1. do **not** commit raw or processed benchmark data;
2. keep `data/` in `.gitignore`;
3. provide official download/source links and preprocessing instructions;
4. cite the original dataset/paper in the manuscript and README;
5. only redistribute files when the dataset license clearly allows it, and include the required license/attribution notices.

## Verified examples

### LargeST

Official repository: https://github.com/liuxu77/LargeST

The official LargeST repository states that the **benchmark dataset is CC BY-NC 4.0** and the code is MIT licensed. Redistribution/adaptation is therefore subject to CC BY-NC 4.0 attribution and non-commercial restrictions. For a clean anonymous repository, linking to the official download and providing preprocessing instructions is still preferable to re-uploading the full dataset.

Recommended citation:

```bibtex
@inproceedings{liu2023largest,
  title={LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting},
  author={Liu, Xu and Xia, Yutong and Liang, Yuxuan and Hu, Junfeng and Wang, Yiwei and Bai, Lei and Huang, Chao and Liu, Zhenguang and Hooi, Bryan and Zimmermann, Roger},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```

### SDWPF

Official dataset page: https://doi.org/10.6084/m9.figshare.24798654

The Figshare dataset page lists **CC BY 4.0**, which permits sharing/adaptation with attribution. If you redistribute a processed copy, retain attribution and clearly state what preprocessing you performed.

Recommended citation:

```bibtex
@article{zhou2024sdwpf,
  title={SDWPF: A Dataset for Spatial Dynamic Wind Power Forecasting over a Large Turbine Array},
  author={Zhou, Jingbo and Lu, Xinjiang and Xiao, Yixiong and Tang, Jian and Su, Jiantao and Li, Yu and Liu, Ji and Lyu, Junfu and Ma, Yanjun and Dou, Dejing},
  journal={Scientific Data},
  volume={11},
  number={1},
  pages={649},
  year={2024}
}
```

### Milan telecommunications data

Dataset paper: https://doi.org/10.1038/sdata.2015.55

Harvard Dataverse dataset DOI: https://doi.org/10.7910/DVN/EGZHFV

The dataset is described as released under the **Open Database License (ODbL)**. ODbL has attribution and database share-alike requirements. Because processed arrays can constitute a derived/adapted database, do not assume that a transformed `.npy` file can be relicensed as your own unrestricted data. Linking to the original dataset and publishing the preprocessing script is the cleaner option.

### LibCity grid files

LibCity code is Apache-2.0 licensed, but that **does not automatically mean every third-party dataset distributed or converted by LibCity has the same license**. Check the original source terms for NYCTaxi, CHIBike, and T-Drive before redistributing the `.grid` files. For an anonymous paper repository, provide acquisition/preprocessing instructions instead of committing the data.

## Datasets whose redistribution terms should be checked before upload

For PEMS03/04/07/08, KnowAir, and any locally processed benchmark files, do not infer redistribution rights from the fact that another GitHub repository hosts a copy. Verify the original dataset/source license first. If the terms are absent or ambiguous, keep the data out of Git and provide a source/download instruction instead.
