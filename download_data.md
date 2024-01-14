To download CODEX spleen dataset from HuBMaP data portal to the directory `data/spleen`:

```bash
cd data
wget -i image_address_spleen.txt -P spleen
```

To continue downloading CODEX lymph node dataset from HuBMaP data portal to the directory `data/lymph_node`:

```bash
wget -i image_address_lymph_node.txt -P lymph_node
```

CODEX large intestine dataset and small intestine dataset cannot directly use `wget` command to download. Please download the data from [HuBMAP](https://portal.hubmapconsortium.org/) and put the data under the folder `data/large_intestine` and `data/small_intestine` respectively.
Here is a list of URLs to download the large intestine data:
https://doi.org/10.35079/HBM334.QWFV.953
https://doi.org/10.35079/HBM353.NZVQ.793
https://doi.org/10.35079/HBM424.STVV.842
https://doi.org/10.35079/HBM462.JKCN.863
https://doi.org/10.35079/HBM622.STKS.394
https://doi.org/10.35079/HBM575.THQM.284
https://doi.org/10.35079/HBM683.NRPR.962
https://doi.org/10.35079/HBM729.XTBN.693
https://doi.org/10.35079/HBM739.HCWP.359
https://doi.org/10.35079/HBM429.LLRT.546
https://doi.org/10.35079/HBM438.JXJW.249
https://doi.org/10.35079/HBM439.WJDV.974
https://doi.org/10.35079/HBM742.NHHQ.357
https://doi.org/10.35079/HBM792.FFJT.499

Here is a list of URLs to download the small intestine data:
https://doi.org/10.35079/HBM443.LGZK.435
https://doi.org/10.35079/HBM466.XSKL.867
https://doi.org/10.35079/HBM666.RBCG.529
https://doi.org/10.35079/HBM284.SBPR.357
https://doi.org/10.35079/HBM676.QVGZ.455
https://doi.org/10.35079/HBM785.FJVT.469
https://doi.org/10.35079/HBM687.SJLD.889
https://doi.org/10.35079/HBM934.KLGL.584
https://doi.org/10.35079/HBM893.MCGS.487
https://doi.org/10.35079/HBM845.VMSZ.536
https://doi.org/10.35079/HBM899.KTQM.246
https://doi.org/10.35079/HBM945.FSHR.864
https://doi.org/10.35079/HBM394.VSKR.883
https://doi.org/10.35079/HBM727.DMKG.675
https://doi.org/10.35079/HBM953.KMTG.758
https://doi.org/10.35079/HBM996.MDQH.988

IMC pancreas dataset can be downloaded from the HTAN data portal follow this URL:
https://humantumoratlas.org/explore?selectedFilters=%5B%7B%22value%22%3A%22IMC%22%2C%22group%22%3A%22assayName%22%2C%22count%22%3A79%2C%22isSelected%22%3Afalse%7D%2C%7B%22value%22%3A%22Pancreas+NOS%22%2C%22group%22%3A%22TissueorOrganofOrigin%22%2C%22count%22%3A78%2C%22isSelected%22%3Afalse%7D%5D&tab=file.
Then put the data under the folder `data/pancreas`.


