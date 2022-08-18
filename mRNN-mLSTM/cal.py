import numpy as np
def get_result(arr):
    return min(arr), np.mean(arr), np.std(arr)

RMSE=[0.26774027556062724, 0.27065384187484753, 0.24501156972291255, 0.2561862168286303, 0.24837076634526517, 0.2515026576005388, 0.24980640678836144, 0.24566599775402823, 0.2815276999820522, 0.24702855664988724, 0.24632712041874827, 0.2564322037033024, 0.2768403023809729, 0.24569798499754444, 0.2722347157425425, 0.2581597031032829, 0.2995661199235854, 0.2459672685071235, 0.2590420625973434, 0.24594779627055482, 0.25342582547624937, 0.26697181927510855, 0.2993504952156417, 0.24477925764059308, 0.29824643447144095, 0.26713041687999217, 0.26507834779659883, 0.26805273460719076, 0.24385458201189086, 0.26646364147111073, 0.24838769466476318, 0.25067738105032294, 0.26977116852360683, 0.2661727314109365, 0.2753731500599298, 0.28206107981776385, 0.26888412293535036, 0.284347430634244, 0.2787239109894098, 0.250863998578694, 0.269940580793335, 0.2585161269717591, 0.24894905254670052, 0.24535449724361916, 0.283151420482881, 0.2404946048111405, 0.26875028416905367, 0.25765416870618196, 0.28482176170309975, 0.30028134130277706, 0.24199846775975487, 0.24684196132666503, 0.2990962711432054, 0.26877787424389143, 0.28751075683938504, 0.24481521006665471, 0.24604320717812428, 0.26433853554764936, 0.286045361900318, 0.24576236349881622, 0.27426640293892524, 0.25302968130021464, 0.24593614176485287, 0.2611901997083429, 0.2442380495587374, 0.2456316012927336, 0.2850139526970522, 0.2760255209319436, 0.2482318343420859, 0.2944703240550588, 0.26611138078589, 0.2989502363737978, 0.2688624260863319, 0.2458330430154664, 0.2463890113222225, 0.27513404916580575, 0.24642724510605518, 0.24459322126596633, 0.27477190769308335, 0.2503608151876309, 0.2850717589513842, 0.24801166817361311, 0.2453731416076669, 0.26621077011396216, 0.2584779748264448, 0.24549703424881383, 0.29999201327135744, 0.25105407468359914, 0.27229825713711464, 0.26828149412404506, 0.2609238578111241, 0.2626049632745083, 0.2431850240520872, 0.24744463294829586, 0.2658028690896784, 0.24688795157854343, 0.29676248873883293, 0.3003384795182079, 0.2693401334379785, 0.27604207333742]
MAE=[0.2107369128643775, 0.21200592919714098, 0.18107879121760656, 0.19513110457189803, 0.18778111543383322, 0.19123937135482777, 0.18684560975082437, 0.18403496983221776, 0.23234176167548895, 0.1854660202578494, 0.18493057956385975, 0.19922899813897457, 0.22169345769026705, 0.18328391365212243, 0.2116487614842514, 0.20205681340607592, 0.2540574745365729, 0.18303724989471562, 0.2024568916243696, 0.183766647720914, 0.19619682632364632, 0.21444095015816367, 0.2541079208930456, 0.18230901673638528, 0.25270440897930646, 0.20414891594460058, 0.20104662612624463, 0.20746313399438088, 0.18060589458825455, 0.20559547969142997, 0.18749480543494834, 0.1917985006691443, 0.211260350761227, 0.20107221451578258, 0.21248362634780255, 0.2339821715878515, 0.2045326141945117, 0.2364443407192974, 0.22673266794924823, 0.19150294197813458, 0.21177553668371923, 0.19804704044142463, 0.187372124673145, 0.18309819793038598, 0.2349557105285445, 0.1705993762556086, 0.2169855691955717, 0.2003111732575522, 0.22969001181196352, 0.2541205422084245, 0.1755197483618786, 0.18644580993000173, 0.25373857716567766, 0.2056049568116337, 0.22615765002889499, 0.1817407873838471, 0.18463450242000928, 0.20936290448753864, 0.23882379359079614, 0.1840209687936434, 0.21825505882535431, 0.193079580615786, 0.18492989332573553, 0.20155518949269186, 0.1800342658633892, 0.18340432815232918, 0.23497531995097654, 0.2217088776167726, 0.18804953620054274, 0.24920518817633636, 0.20603032969208954, 0.2534152738029389, 0.2097519637731522, 0.18400639050139622, 0.18532478950946538, 0.22389741212526024, 0.18462999300540966, 0.18147214909454318, 0.22355335645033317, 0.19096086720194702, 0.23718802102539593, 0.18817619606063601, 0.18324647704350752, 0.20312598459632478, 0.19331019497145707, 0.18304858968320645, 0.25457354886895456, 0.18744399528543673, 0.22004849918572397, 0.21558120191183372, 0.20450401827114822, 0.20714481422140107, 0.17801784197415657, 0.18640927218722228, 0.21220049660035548, 0.1860186693180158, 0.2515465401618393, 0.25429750330787326, 0.21615679595582046, 0.21246794167689947]
MAPE=[1.1881756622194635, 1.0373708409336957, 1.8381559821102038, 1.2291165532524753, 1.699395872387088, 1.2039243372243846, 1.9933739046859495, 1.7166702449225648, 1.257158129394825, 1.7513969111389203, 1.704679139962728, 1.8058856639081795, 1.043978789204167, 1.6845234753475553, 1.0178338656719483, 1.308339049131826, 1.7349782335311477, 1.835717848951831, 1.4750546941348652, 1.7772826958905825, 1.4141456756630482, 1.1829004997105927, 1.738511618981606, 1.695989254413222, 1.6938513207453423, 1.5553693510805309, 1.2276142994429464, 1.1103359152968841, 1.6895614277437303, 1.1930410571838332, 1.7235081722770562, 1.7929260076851739, 1.0677205091956083, 1.1899333371092111, 1.043697415888611, 1.3238381458725998, 1.1057285216646529, 1.353775442510362, 1.2596478386156758, 1.6711459105429567, 1.065209438614198, 1.2873104910639788, 1.6668508487242326, 1.7440189270638802, 1.3421560537548332, 1.9020097753791296, 1.1798935693204577, 1.4547140373572076, 1.2803348971746162, 1.7654252625649625, 1.8041421415007879, 1.6629825450523952, 1.7270802569534207, 1.1020639598047954, 1.3133004435810824, 1.7722529653365493, 1.730658843824467, 1.2554125663924902, 1.397385675567831, 1.7201446791652777, 0.9905309339025218, 1.6399486532768999, 1.6730455864046752, 1.3418667492353846, 1.65299161276618, 1.7612990272490203, 1.3059740731692826, 1.0230085431550213, 1.6683829677937143, 1.584350756222662, 1.4275358052751106, 1.7190053598813086, 1.1048123309501092, 1.7427831915940861, 1.6580914741038257, 1.2265639970220816, 1.7833863366512224, 1.7401766245191916, 1.1779318640272165, 1.7828766186623535, 1.3473861049736058, 1.6999145083066691, 1.7043333503168903, 1.1681932138916284, 1.5206156692863861, 1.739752320937282, 1.7632391721337612, 2.1519204447523954, 1.175497344860807, 1.2062740252641455, 1.3080779521811579, 1.366313223233262, 1.6957313532587817, 1.6879186471733068, 1.20260041309151, 1.7414165651567743, 1.656353918742259, 1.7694451622552236, 1.1931499038741928, 1.0607883933424511]

print(get_result(RMSE))
print(get_result(MAE))
print(get_result(MAPE))