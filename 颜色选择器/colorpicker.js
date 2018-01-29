/*在html中放置
1、<div id="_colorset" ></div>空标签
2、直接使用_value获取得到的值
3、_color是每个颜色的类
4、由于使用js设定的样式所以有的样式css不能设置，可以使用js在此
   文件后面修改
5、选中后自动隐藏。
6、css中设定_colorset可以初始隐藏
*/





var  _value=""
function colorpicker()
{
	colorlist=["E5006A","E5004F","FF0000","EB6100","F39800","FCC800","FFF100","CFDB00","8FC31F","22AC38","009944","009B6B","009E96","00A0C1","00A0E9","0086D1","0068B7","00479D","1D2088","601986","920783","BE0081","FFF","000"]
	var ob1=document.getElementById("_colorset")

	{
		ob1.style.border="1px solid #CCC"
		ob1.style.width=174+"px"
		ob1.style.height=117+"px"
	}
	
	for(var i=0;i<24;i++)
	{   
		var node=document.createElement("input")
		node.type="button"
		node.cl=""
		node.class="_clor"
		ob1.append(node)
		var ob=ob1.getElementsByTagName("input")

		{
			ob[i].style.float="left"
			ob[i].style.borderStyle="none"
			ob[i].style.width=25+"px";
			ob[i].style.margin=2+"px";
			ob[i].style.height=25+"px";  
			ob[i].style.cursor="pointer";
			ob[i].style.border="1px solid #ccc";
		}

		 (function(j){
			ob[j].style.background="#"+colorlist[i]
			ob[j].cl="#"+colorlist[i]
			ob[j].onclick=function(e)
			{
				_value=this.cl
				ob1.style.display="none"
			}
		})(i)
	}
	
}
window.onload=function()
{
		colorpicker()
}

