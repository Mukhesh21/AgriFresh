# AgriFresh: Dataset Analysis & Inspection Report

**Dataset**: AgriFreshNET Freshness and Shelf-Life Image Dataset  
**Total Image Count**: 14,160  
**Valid Images**: 14,160 (100.00%)  
**Corrupted Images**: 0  
**Duplicate Images Detected**: 40  

## 1. Summary Statistics

| Metric | Value |
| :--- | :--- |
| **Dataset Path** | D:\ML Project Freshness Detection\AgriFreshNET Freshness and Shelf-Life Image Datase\AgriFreshNET Freshness and Shelf-Life Image Datase\Processed Data\Processed Data |
| **Total Folders** | 24 |
| **Total Images Found** | 14160 |
| **Valid Images** | 14160 |
| **Corrupted Images** | 0 |
| **Duplicate Images** | 40 |
| **Fresh Images** | 4720 |
| **Semi-Fresh Images** | 4720 |
| **Rotten Images** | 4720 |
| **Produce Types Count** | 8 |
| **Min Width** | 512 |
| **Max Width** | 512 |
| **Mean Width** | 512.0 |
| **Min Height** | 512 |
| **Max Height** | 512 |
| **Mean Height** | 512.0 |
| **Primary Color Modes** | RGB: 14160 |
| **Primary Image Format** | JPEG: 14160 |

## 2. Freshness Class Distribution

| Freshness Class | Class Index | Image Count | Proportion (%) |
| :--- | :---: | :---: | :---: |
| **Fresh** | `0` | 4,720 | 33.33% |
| **Semi-Fresh** | `1` | 4,720 | 33.33% |
| **Rotten** | `2` | 4,720 | 33.33% |

## 3. Freshness by Produce Type Breakdown

| produce     |   Fresh |   Rotten |   Semi-Fresh |   All |
|:------------|--------:|---------:|-------------:|------:|
| Banana      |     590 |      590 |          590 |  1770 |
| Bittermelon |     590 |      590 |          590 |  1770 |
| Cucumber    |     590 |      590 |          590 |  1770 |
| Eggplant    |     590 |      590 |          590 |  1770 |
| Orange      |     590 |      590 |          590 |  1770 |
| Papaya      |     590 |      590 |          590 |  1770 |
| Pineapple   |     590 |      590 |          590 |  1770 |
| Tomato      |     590 |      590 |          590 |  1770 |
| All         |    4720 |     4720 |         4720 | 14160 |


## 4. Folder-Level Analysis (24 Folders)

| Folder Name | Freshness Class | Produce Type | Shelf-Life Range | Image Count | Valid | Corrupt |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `Fresh Banana(1-4)` | Fresh | Banana | 1-4 | 590 | 590 | 0 |
| `Fresh Bittermelon(1-3)` | Fresh | Bittermelon | 1-3 | 590 | 590 | 0 |
| `Fresh Cucumber(1-6)` | Fresh | Cucumber | 1-6 | 590 | 590 | 0 |
| `Fresh Orange(1-9)` | Fresh | Orange | 1-9 | 590 | 590 | 0 |
| `Fresh Papaya(1-4)` | Fresh | Papaya | 1-4 | 590 | 590 | 0 |
| `Fresh Tomato(1-10)` | Fresh | Tomato | 1-10 | 590 | 590 | 0 |
| `Fresh eggplant(1-4)` | Fresh | Eggplant | 1-4 | 590 | 590 | 0 |
| `Fresh pineapple(1-15)` | Fresh | Pineapple | 1-15 | 590 | 590 | 0 |
| `Rotten Bittermelon(5-8)` | Rotten | Bittermelon | 5-8 | 590 | 590 | 0 |
| `Rotten Cucumber(12-20)` | Rotten | Cucumber | 12-20 | 590 | 590 | 0 |
| `Rotten Orange(20-35)` | Rotten | Orange | 20-35 | 590 | 590 | 0 |
| `Rotten Papaya(7-12)` | Rotten | Papaya | 7-12 | 590 | 590 | 0 |
| `Rotten Pineapple(25-35)` | Rotten | Pineapple | 25-35 | 590 | 590 | 0 |
| `Rotten Tomato(24-35)` | Rotten | Tomato | 24-35 | 590 | 590 | 0 |
| `Rotten banana(7-13)` | Rotten | Banana | 7-13 | 590 | 590 | 0 |
| `Rotten eggplant(8-15)` | Rotten | Eggplant | 8-15 | 590 | 590 | 0 |
| `Semi Fresh Bittermelon ( 3-5)` | Semi-Fresh | Bittermelon | 3-5 | 590 | 590 | 0 |
| `Semi Fresh Cucumber(6-12)` | Semi-Fresh | Cucumber | 6-12 | 590 | 590 | 0 |
| `Semi Fresh Papaya(4-7)` | Semi-Fresh | Papaya | 4-7 | 590 | 590 | 0 |
| `Semi fresh Orange(9-20)` | Semi-Fresh | Orange | 9-20 | 590 | 590 | 0 |
| `Semi fresh Pineapple (15-25)` | Semi-Fresh | Pineapple | 15-25 | 590 | 590 | 0 |
| `Semi fresh Tomato(10-24)` | Semi-Fresh | Tomato | 10-24 | 590 | 590 | 0 |
| `Semi fresh banana(4-7)` | Semi-Fresh | Banana | 4-7 | 590 | 590 | 0 |
| `Semi_Fresh eggplant(4-8)` | Semi-Fresh | Eggplant | 4-8 | 590 | 590 | 0 |

## 5. Data Quality & Preprocessing Assessment

- **Corruption Status**: No unreadable/truncated image headers found.
- **Color Space**: Uniform RGB representation observed across images.
- **Resolution Consistency**: Resolutions vary across raw captures (Min: 512x512, Max: 512x512). Standardization to **224x224** via bilinear interpolation is recommended for ImageNet backbone compatibility.
- **Class Imbalance**: Class distribution is reasonably balanced across Fresh, Semi-Fresh, and Rotten categories, but stratified splitting is required to ensure even produce representation across splits.