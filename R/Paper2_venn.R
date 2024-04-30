library (VennDiagram)  
library(openxlsx)

#数值导入，可对数值进行配对
#devign
#set4<-read.xlsx('C:/Users/Senlei Xu/Downloads/venn_devign_data.xlsx',sheet= "Sheet4",sep=',')
#set5<-read.xlsx('C:/Users/Senlei Xu/Downloads/venn_devign_data.xlsx',sheet= "Sheet5",sep=',')
#set6<-read.xlsx('C:/Users/Senlei Xu/Downloads/venn_devign_data.xlsx',sheet= "Sheet6",sep=',')

#reveal
#set4<-read.xlsx('C:/Users/Senlei Xu/Downloads/venn_reveal_data.xlsx',sheet= "Sheet1",sep=',')
#set5<-read.xlsx('C:/Users/Senlei Xu/Downloads/venn_reveal_data.xlsx',sheet= "Sheet2",sep=',')
#set6<-read.xlsx('C:/Users/Senlei Xu/Downloads/venn_reveal_data.xlsx',sheet= "Sheet3",sep=',')

#fan
#set4<-read.xlsx('C:/Users/Senlei Xu/Downloads/venn_fan_data.xlsx',sheet= "Sheet1",sep=',')
#set5<-read.xlsx('C:/Users/Senlei Xu/Downloads/venn_fan_data.xlsx',sheet= "Sheet2",sep=',')
#set6<-read.xlsx('C:/Users/Senlei Xu/Downloads/venn_fan_data.xlsx',sheet= "Sheet3",sep=',')

#combined
set4<-read.xlsx('C:/Users/Senlei Xu/Downloads/venn_combined_data.xlsx',sheet= "Sheet1",sep=',')
set5<-read.xlsx('C:/Users/Senlei Xu/Downloads/venn_combined_data.xlsx',sheet= "Sheet2",sep=',')
set6<-read.xlsx('C:/Users/Senlei Xu/Downloads/venn_combined_data.xlsx',sheet= "Sheet3",sep=',')


#数据转置，如果不转后头函数venn.diagram对矩阵数值不识别#
set4=t(set4)
set5=t(set5)
set6=t(set6)

#三元#
venn.diagram(x=list(set4,set5,set6),
             scaled = F, # 根据比例显示大小
             alpha= 0.5, #透明度
             lwd=1,lty=1,col=c('#F0F059','#3EF0F0','#F78888'), #圆圈线条粗细、形状、颜色；1 实线, 2 虚线, blank无线条
             label.col ='black' , # 数字颜色label.col=c('#FFFFCC','#CCFFFF',......)根据不同颜色显示数值颜色
             cex = 2, # 数字大小
             fontface = "bold",  # 字体粗细；加粗bold
             fill=c('#F0F059','#3EF0F0','#F78888'), # 填充色 配色https://www.58pic.com/
             category.names = c("GraphCodeBERT", "ContraBERT","SCL-CVD") , #标签名
             cat.dist = 0.01, # 标签距离圆圈的远近
             cat.pos = c(-10, 10, 175), # 标签相对于圆圈的角度cat.pos = c(-10, 10, 135)
             cat.cex = 2, #标签字体大小
             cat.fontface = "bold",  # 标签字体加粗
             cat.col='black' ,   #cat.col=c('#FFFFCC','#CCFFFF',.....)根据相应颜色改变标签颜色
             cat.default.pos = "outer",  # 标签位置, outer内;text 外
             output=TRUE,
             filename='C:/Users/Senlei Xu/Downloads/venn_combined.png',# 文件保存
             imagetype="png",  # 类型（tiff png svg）
             resolution = 400,  # 分辨率
             compression = "lzw"# 压缩算法
)
