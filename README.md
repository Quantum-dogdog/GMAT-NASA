## GMAT-NASA
此脚本适用于NASA的General Mission Analysis Tool（GMAT）开源软件，该软件的中文名可以翻译为 通用任务分析工具，还是值得深入玩一玩的。
关于这个软件，网络上公开的学习资料较少，在这里分享一个我自己做的EMJ+DSM（从地球近地轨道发射，经过一次火星重力辅助，中间施加两次深空机动，最终进入木星大椭圆轨道）的行星际转移算例。

## 1

之所以叫Simulate Juno Probe，是因为该航天器的有效载荷使用的是Juno的有效载荷1593kg。
该航天器的初始状态是停泊在在倾角20°的400km（6378+400）近地圆轨道上，因为现实中的航天器大多采用甲基肼+四氧化二氮（MMH/NTO）作为真空燃料，所以脚本中燃料密度用的是甲基肼的密度880kg/m3，用的引擎是YF-50D的数据，推力6.5kN，Isp316s，然后发射日期是2031年2月14日。
注意：本算例不是最优轨迹（算例中的航天器的燃料罐足足有270吨，进入木星大椭圆轨道后仅剩余<50kg燃料），主要是给大家提供在GMAT中实现引力弹弓的参考模板。



## 2024-02-03更新

## 2

新上传了模拟天问4号的过程中产生的5个脚本，给出了天问4号的另外一种可能（现实中的天问4号将采用EVEEJ飞行序列，这里采用的是EEJ飞行序列），与现实中的天问4号不同的地方主要有：1.发射日期不同（现实中大概是2029年9月，这里是2025年9月）；2.探测器质量不同（现实中大概是5t，这里的两个算例分别是2.8t、5t）；3.地球停泊轨道和木星捕获轨道不同（现实中停泊轨道大概是LEO，捕获轨道大概是69911+70000km，这里停泊轨道大约是6378+4000km，捕获轨道大约是69911+150000km）。

## 3

Tianwen4 arrive at 2032这个算例是从LEO6378+700km出发的，只需要你用火箭发射12t到地球停泊轨道上；文件名中括号里的质量指的是天问4号实施完JOI减速，被木星捕获后剩余的质量，其中1601kg需要你用火箭发射17t到地球停泊轨道上，2593kg需要你用火箭发射31t到地球停泊轨道上（当下没有火箭可以做到）。

## 4

![image](https://github.com/Quantum-dogdog/GMAT-NASA/blob/main/eej.jpg)
根据NASA Ames Research Center Trajectory Browser（ https://trajbrowser.arc.nasa.gov/traj_browser.php ）里User Guide页面下最后一段的说法，Trajectory Browser是一个低保真轨道设计工具，GMAT是一个高保真轨道设计工具，这是因为Trajectory Browser只考虑太阳的引力，而GMAT可以考虑所有行星的摄动、光压、球谐重力球......所以你是没有办法完全复现Trajectory Browser里得出的发射日期、deltaV的（事实上我们这个模拟的算例中的发射日期比Trajectory Browser给出的2025.10.20早大约1个月，深空机动比Trajectory Browser给出的0.724m/s大40%）。Trajectory Browser里是从250km的LEO出发，deltaV是4.3km/s，我试了很多次，都达不到这个数，但是用GTO作为停泊轨道的话，是可以达到这个数的。

## 5

Tianwen4括号里带质量的两个算例的任务时间是和Trajectory Browser的4.77年十分接近的，但是发射日期、飞掠日期、到达日期均和Trajectory Browser有误差；Tianwen4 arrive at 2032的算例的发射时间和飞掠时间同Trajectory Browser一样，但是到达时间比Trajectory Browser的2030年晚2年，所以是无法完全复现Trajectory Browser这个低保真轨道计算工具给出的答案的，所以不用钻牛角尖，非要一模一样。

## 6

后续的提升：算例中的机动都是瞬时完成的，而现实中都是finiteburn，所以大家有兴趣的话可以尝试将算例中的impulsiveburn升级成finiteburn，那样finiteburn就需要同时考虑推力和Isp了，要达到同样的C3，同样的Isp，不同的推力消耗的燃料不同。


## 2024-03-07更新

## 7

新增了finiteburn到GEO、finiteburn模拟嫦娥5号任务的脚本（该模拟基本做到了与现实世界同时序（即2020.11.24 04:30（北京时间，比UTC时间早8小时）+540s二级一次点火（根据发射直播截图，此时的经纬度和高度大约是东经126.24，北纬15.85,187.062千米），+740s二级一次关机，+1677s二级二次点火，+2114s二级二次关机），但是抵达月球的轨道参数没能完全与现实一样）。

## 2024-04-29更新

## 8

新增了QueQiao from earth to moon，模拟鹊桥3号任务（平行世界中今年3月28日发射），该模拟采用月球引力辅助转移轨道进驻Halo Orbit. TLI deltaV是3.02km/s量级,近月点机动300m/s量级（《地月拉格朗日L2点中继星轨道分析与设计》论文中最优是200m/s量级，当然了本算例中的Halo Orbit轨道参数与论文不同）,Halo Orbit 插入机动50m/s量级。


## 2024-05-04更新

## 9

新增了 halo orbit and invariant manifolds脚本，模拟围绕地月L2拉格朗日点的晕轨道。

## 10

新增了嫦娥6号任务模拟，本次模拟是从二级二次关机、末速修正、器箭分离那个时间点开始，根据模拟来看，嫦娥6号大概在北京时间5月8日11点16分到达月球（不一定准，只是我的猜测，猜不对不负责）。然后在月球近点实施减速机动，由于官方未给出大椭圆轨道参数，在此设置了3次300m/s的减速机动，相应的大椭圆轨道远点高度分别为30000、7000，模拟中采用的嫦娥6号的质量为8t，进入使命轨道后剩余质量为5.9t.



## 2024-05-12更新

## 11

新增了嫦娥6号任务模拟马后炮版本，在5月4号至12号这个时间段内，我通过观看更多视频获得了月球大椭圆轨道的周期参数：12h、4h，获得了现实世界中嫦娥6号抵达月球时间（5月8日10时12分），从而倒推了轨道六根数。整个模拟经历告诉我们：仿真终究不是现实，不是一锤子买卖，你可以在现实中参考仿真的数据，但是它是理想的，而现实总会存在误差，需要把现实中实际的误差动态考虑进去，时刻仿真才行。

## 12

新增了发射窗口计算器demo版本算例，该算例是以去火星为例子，展示了GMAT Optimizer的用法，该算例与官方例子中的发射窗口计算器相比，更易理解。因为仅做演示，发射窗口时间选择范围为2025年1月4日-6日，飞行时间选择范围为200天或210天，然后计算相应的发射deltaV进行比较这样子。该算例采用的是GMAT自带的Yukon Optimizer,大家也可以用Thinking System Inc.官网上的VF13ad Optimizer第三方插件，一样好用。
PS:还有更强大的libsnopt Optimizer第三方插件，但是因为snopt是商用的，所以这个第三方插件是搞不到的，此外经过我的实验，使用snopt试用版本的.dll也无法自行构建成功，供大家参考，节省力气。

## 2024-09-26更新

## 13

新增了dro.script,远距离月球逆行轨道的示例（远距离逆行是在地月会合坐标系下围绕月球画圈圈），月球惯性坐标系下周期是10.69天，在地月惯性坐标系下的周期是30.52天，也就是1:1共振轨道，主要考虑也只需要考虑太阳、地球、月球的引力以及太阳光压，因为其它摄动力的影响较小，具体可见《DRO计算及其在地月系中的摄动力研究》论文。

## 14

新增EarthMoonFreeReturn.script，实现的近月点是13000km（含月球半径），返回时近地点10000km（含地球半径），地月自由返回轨道允许航天器从地球出发到达月球，在近月点无需施加额外推力，即可自动返回地球（8字形，即四个地月转移轨道之一的地球顺形月球逆行轨道）。

## 2024-10-02更新

## 15

![image](https://github.com/Quantum-dogdog/GMAT-NASA/blob/main/periapsis%20's%20location.png)

一张直观、明晰的示意图，清楚的告诉你：如果你想要Vinf与地球公转速度平行时，近地点（即施加deltaV的点）与日地连线的关系。

以及两个分析的脚本文件：fenxi_dianhuoshiji.script & fenxi2_1bi2gongzhenguidao.script


## 2024-10-28更新

## 16

新增平行世界的EuropaClipper.script,飞行时序、飞掠时间与现实几乎完全一致，主要参考NASA的eyes on the solar system网页可视化工具中的数据进行仿真，MMH/NTO Thruster的官方数据未查到（采用的是[rocketCEA](https://github.com/sonofeft/RocketCEA)里理想1维状态下的最高比冲350s），可惜并不掌握接口抓包的技能（希望搞小猿搜题算法大赛的人中也有航天爱好者，能来抓一下NASA的包），而且eyes on the solar system也不给出深空机动的计划时间，所以只能根据自己喜好选定几个DSM的时刻，导致2750kg的燃料不够用（需要-370kg）。据此，给此次轨道仿真只打81分。

## 2025-01-14更新

## 17

新增jupiter system.script,教给大家怎么导入SPICE kernel来添加木星系的4大卫星（指Io、Europa、Ganymede、Callisto），学会了可以推广到添加土星卫星。首先你需要在naif.jpl.nasa.gov上获取jup280.bsp或者jup344.bsp或者jup365.bsp（其实de421.bsp、de430.bsp、de440.bsp和他们有对应的版本关系，但是不知为何在GMAT2022a中de421搭配jup365并不报错,在GMAT2018a中就会报错），网站上不去可以到spiftp.esac.esa.int或者data.darts.isas.jaxa.jp，esa和jaxa做的镜像。然后你需要下载卫星的global map来做贴图（如果你想直接用3d模型的话，需要在3dsmax2009中把网上的资源转换成.3ds格式才能在GMAT中显示），我在这里配套传了Io.jpg和Europa.jpg。卫星的参数的设置需要看ssd.jpl.nasa.gov/sats/phys_par以及Report of the IAU Working Group on cartographic coordinates and rotational elements: 2009这篇论文（为什么用IAU2009是因为jup365与它协调，教程里添加土卫六泰坦的数据来自IAU2006，然后这个数据说是每3年更新1次，但是好像因为IAU2015精度搞得很高，这几年都没有更新的）。
