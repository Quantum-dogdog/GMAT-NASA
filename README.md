# GMAT-NASA
此脚本适用于NASA的General Mission Analysis Tool（GMAT）开源软件，该软件的中文名可以翻译为 通用任务分析工具，还是值得深入玩一玩的。
关于这个软件，网络上公开的学习资料较少，在这里分享一个我自己做的EMJ+DSM（从地球近地轨道发射，经过一次火星重力辅助，中间施加两次深空机动，最终进入木星大椭圆轨道）的行星际转移算例。

#1

之所以叫Simulate Juno Probe，是因为该航天器的有效载荷使用的是Juno的有效载荷1593kg。
该航天器的初始状态是停泊在在倾角20°的400km（6378+400）近地圆轨道上，因为现实中的航天器大多采用甲基肼+四氧化二氮（MMH/NTO）作为真空燃料，所以脚本中燃料密度用的是甲基肼的密度880kg/m3，用的引擎是YF-50D的数据，推力6.5kN，Isp316s，然后发射日期是2031年2月14日。
注意：本算例不是最优轨迹（算例中的航天器的燃料罐足足有270吨，进入木星大椭圆轨道后仅剩余<50kg燃料），主要是给大家提供在GMAT中实现引力弹弓的参考模板。



#2024-02-03更新

#2

新上传了模拟天问4号的过程中产生的5个脚本，给出了天问4号的另外一种可能（现实中的天问4号将采用EVEEJ飞行序列，这里采用的是EEJ飞行序列），与现实中的天问4号不同的地方主要有：1.发射日期不同（现实中大概是2029年9月，这里是2025年9月）；2.探测器质量不同（现实中大概是5t，这里的两个算例分别是2.8t、5t）；3.地球停泊轨道和木星捕获轨道不同（现实中停泊轨道大概是LEO，捕获轨道大概是69911+70000km，这里停泊轨道大约是6378+4000km，捕获轨道大约是69911+150000km）。

#3

Tianwen4 arrive at 2032这个算例是从LEO6378+700km出发的，只需要你用火箭发射12t到地球停泊轨道上；文件名中括号里的质量指的是天问4号实施完JOI减速，被木星捕获后剩余的质量，其中1601kg需要你用火箭发射17t到地球停泊轨道上，2593kg需要你用火箭发射31t到地球停泊轨道上（当下没有火箭可以做到）。

#4

![image](https://github.com/Quantum-dogdog/GMAT-NASA/blob/main/eej.jpg)
根据NASA Ames Research Center Trajectory Browser（ https://trajbrowser.arc.nasa.gov/traj_browser.php ）里User Guide页面下最后一段的说法，Trajectory Browser是一个低保真轨道设计工具，GMAT是一个高保真轨道设计工具，这是因为Trajectory Browser只考虑太阳的引力，而GMAT可以考虑所有行星的摄动、光压、球谐重力球......所以你是没有办法完全复现Trajectory Browser里得出的发射日期、deltaV的（事实上我们这个模拟的算例中的发射日期比Trajectory Browser给出的2025.10.20早大约1个月，深空机动比Trajectory Browser给出的0.724m/s大40%），所以不用钻牛角尖，非要一模一样。

#5

Tianwen4括号里带质量的两个算例的任务时间是和Trajectory Browser的4.77年十分接近的，但是发射日期、飞掠日期、到达日期均和Trajectory Browser有误差；Tianwen4 arrive at 2032的算例的发射时间和飞掠时间同Trajectory Browser一样，但是到达时间比Trajectory Browser的2030年晚2年，所以是无法完全复现Trajectory Browser这个低保真轨道计算工具给出的答案的。

#6

后续的提升：算例中的机动都是瞬时完成的，而现实中都是finiteburn，所以大家有兴趣的话可以尝试将算例中的impulsiveburn升级成finiteburn。
