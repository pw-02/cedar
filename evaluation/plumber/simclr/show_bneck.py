import tensorflow as tf

filename = "stats.pb"
plumber = tf.data.experimental.analysis.PlumberPerformanceModel(filename)
model = plumber.model()

recommendation = model.recommendation()
slowest_node = recommendation.bottleneck_node()
print("Slowest node: {}".format(slowest_node.name))
CPU_time_used = model.total_CPU_time()
wallclock_used = model.total_wallclock_time()
cpu_util = model.CPU_Util()
disk_util = model.Disk_Util()
print("Resource utilization: CPU Util {} ({}s CPU time,{}s wallclock time),"
      " Disk Util {}".format(cpu_util,
                        CPU_time_used,
                        wallclock_used,
                        disk_util))