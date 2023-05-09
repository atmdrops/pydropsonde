# Performance of sondes

```{note}
This page explains the context behind the `status` tool, which provides information about the performance of the sondes.
```

After a research flight, your mission PI comes up and asks you for a quick status of the dropsondes' performance, because they want to include it in the flight report. Well, `status` will help you out here... Or if you need to send a list of failed sondes to Vaisala, this tool will help you out again.

----
## Sonde failures

An important role of `status` is to detect and list out sondes that failed. Sonde failures could be because of multiple reasons:

- {ref}`Launch-detect failure <launch-detect-failure>` (exceptionally high during EUREC$^4$A)
- {ref}`Sensor failure <sensor-failure>` (any combination of the `p`, `T`, `RH`, `GPS` sensors could fail)
- {ref}`Parachute failure <parachute-failure>` (parachute doesn't open resulting in fast fall)
- {ref}`Others <other-failure>` (such as initialization failure)

(launch-detect-failure)=
### Launch-detect failure

A launch should be detected automatically in the ideal scenario. If the sonde fails to do this, it does not switch to high-power signal transmission, and thus, stops sending data back to the AVAPS PC, after a short range. 
    
The primary method to check if a launch was detected for a given sonde is to parse through the log files of the sonde. These files have names starting with 'A' and are followed by the date and time of launch. The file extension is the number of the channel used to initialise the sonde and receive its signal, but for all practical purposes, it is a `.txt` file. (Note: For sondes that did not detect a launch, the file name has time when the sonde was initialised). The log file contains an internal record termed 'Launch Obs Done?'. If this value is 1, the launch was detected, else if it is 0, launch was not detected. The same values are used to mark the `ld_FLAG`.

To get a quick idea of how to get sondes that have a launch-detect failure, check out this {doc}`how-to guide <../howto/list_failed_sondes>`

(sensor-failure)=
### Sensor failure

(parachute-failure)=
### Parachute failure

(other-failure)=
### Others
