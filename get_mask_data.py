from human36m.xml import xml
import xmltodict

aaa = xmltodict.parse(xml)
joints_bvh = aaa['skel_angles']['tree']['item']






print(aaa)
