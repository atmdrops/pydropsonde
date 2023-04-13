# Directory Structure

`halodrops` strongly recommends that the data be stored in the following structure:

```
├── Data_Directory
│   ├── 20200202
│   │   ├── Level_0
│   │   ├── Level_1
│   │   ├── Level_2
│   │   ├── Level_3
│   │   ├── Level_4
│   ├── 20220303
│   │   ├── Level_0
│   │   ├── Level_1
│   │   ├── Level_2
│   │   ├── Level_3
│   │   ├── Level_4
```

## Descriptions - Directory and nomenclature

The `Data_Directory` is a directory that includes all data from a single campaign. Therein, data from individual flights are stored in their respective folders with their name in the format of `YYYYMMDD`, indicating the flight-date. In case of flying through midnight, consider the date of take-off. In case, there are multiple flights in a day, consider adding alphabetical suffixes  to distinguish chronologically between flights, e.g. `20200202-A` and `20200202-B` would be two flights on the same day in the same order that they were flown. 

The directories within flight-data directories with `Level_` prefixes include the different processed levels of data as described below.

| Level     | Description                                                                                    |
| --------- | ---------------------------------------------------------------------------------------------- |
| `Level_0` | Contains all raw files from a flight                                                           |
| `Level_1` | Files generated from the ASPEN-processing of the D-files in Level-0                            |
| `Level_2` | Sounding files that passed additional QC tests, and exclude all soundings with no usable data  |
| `Level_3` | All sounding files in Level-2 gridded on a uniform, vertical grid, with some derived variables |
| `Level_4` | Circle products from all circles flown during flight                                           |

Maintaining this structure is also important for the package to automatically navigate through data directories, e.g. see the {py:obj}`Paths <halodrops.helper.paths.Paths>` class.