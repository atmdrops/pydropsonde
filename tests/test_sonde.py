from halodrops import sonde

s_id = "test_this_id"
launch_time = "2020-02-02 12:30:12"
data = ["placeholder","data",42]

def test_Sonde_attrs():

    TestSonde_nolaunchtime = sonde.Sonde(s_id)
    TestSonde_withlaunchtime = sonde.Sonde(s_id,launch_time=launch_time)

    assert TestSonde_nolaunchtime.serial_id == s_id
    assert TestSonde_nolaunchtime.launch_time is None
    assert TestSonde_withlaunchtime.serial_id == s_id
    assert TestSonde_withlaunchtime.launch_time == launch_time

def test_SondeData_attrs():

    TestSonde_nolaunchtime = sonde.SondeData(s_id,data)
    TestSonde_withlaunchtime = sonde.SondeData(s_id,launch_time=launch_time,data=data)

    assert TestSonde_nolaunchtime.serial_id == s_id
    assert TestSonde_nolaunchtime.launch_time is None
    assert TestSonde_nolaunchtime.data == data
    assert TestSonde_withlaunchtime.serial_id == s_id
    assert TestSonde_withlaunchtime.launch_time == launch_time
    assert TestSonde_withlaunchtime.data == data