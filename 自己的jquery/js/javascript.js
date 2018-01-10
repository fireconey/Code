
	/*
	使用$包装可以使样子与jquery非常像。
	核心类是chajian()，获取的所有对象
	都放在this.element中，插件要调用
	对象可以直接在插件中使用this.element[XX]。
	return this 是返回插件这个对象，这
	样可以使用连号继续调用其中的方法，
	起到连缀的效果（如：a.b.c）。
	*/

	var $= function(x) 
	{  
		var _init = new chajian()
		//使用多级选择器获取对象的并且会放在this.element中
		if (x != null ) 
		{
			_init.query(x)
		}
		return _init
	}









	/*****************定义的一个类*****************/
/*
把自己的函数插入可以调用
$().extend(fn) fn不要括号
*/
function chajian()
{

	//随便弄几个变量以便放值
	this.element = []
	this.temp = 1
	/*
	由于插入操作中innerHTML渲染要
	从新定位元素后才能继续渲染否则
	只渲染最后一次，所以要从新加载
	选择器,_x是暂时保存查找的内容 
	*/
	var _x




	/************************通过类名得到对象*********************/
	this.getClassName = function(x) 
	{
		this.element = []
		var classname = document.getElementsByTagName("*")

		for (var i = 0; i < classname.length; i++)
		{
			if (classname[i].className == x) 
			{

				this.element.push(classname[i])
				this.temp = classname[i]

			}

			if (i == classname.length - 1 && this.element.length == 0) 
			{
				alert("查询类的时候发生错误，是否$()中输入有错?")
			}
		}
		return this
	}





	/*********************通过id得到对象*****************************/
	this.getId = function(x) 
	{   

		var id = document.getElementById(x)
		if (id != null)
		{
			this.element[0] = id
			this.temp = id
		}
		if (id == null) 
		{
			alert("查询id时发生错误，是否$()中输入有错?")
		}
		return this
	}





	/**********************通过元素得到对象************************/
	this.getElement = function(x) 
	{
		var el = document.getElementsByTagName(x)
		if (el.length != 0) 
		{
			this.element = el
			this.temp = el
		}
		if (el.length == 0) 
		{
			alert("查询元素时发生错误，是否$()中输入有错?")
		}
		return this
	}






	/*******通过调用上面其中某个方法后得到一个对象，根据此对象得到子对象***********/

	this.findchild = function(y) 
	{
		y = y.toLowerCase();
		this.temp = [];
		var ale = y;
		var start = y.charAt(0);
		var flag = 1;

		switch (start) 
		{
			case ".":
			y = y.slice(1);
			flag = "className";
			break;
			case "#":
			y = y.slice(1);
			flag = "id";
			break;
			default:
			y = y;
			flag = "tagName";
			break
		}


		for (var i = 0; i < this.element.length; i++) 
		{
			var ele = this.element[i].children

			this.child -= 1
			var ko = this.child
				/*某对象没有子元素的情况下看看
				传入的参数是否还有（参数个数倒
				减计算的形式）是这种的情况就警告*/
				if (ele.length == 0 && ko != 0) 
				{
					alert("父级已经没有子级了！" + "\n报错级数为：" + ale + "\n在findchild函数中报错")
					return;
				}

				for (var j = 0; j < ele.length; j++) 
				{
					var ch = ele[j][flag].toLowerCase()
					if (ch == y) 
					{
						this.temp.push(ele[j])
					}

				/*某级还有子元素，但是下一个参数
				和子元素不一样导致temp空有子元素
				且temp为空，可以确定上面条件。*/
				if (this.temp.length == 0 && j == ele.length - 1) 
				{
					alert("没有此对象:" + ale + "\n或者Html中嵌套错误" + "\n在findchild函数中报错")
					return;
				}
			}
		}
		this.element = this.temp
		return this
	}






	/*****************多级选择*****************/
	this.child = 0 //数据由query修改传给findchild，用于计数有多少个参数传入
	this.query = function(x) 
	{   _x=x    //在操作插入一次中要求更新，所以后面要调用这个函数参数一样
		var q = this.tool(x)
		var tem = 0

		/*
		刚开始通过查看第传入的第一个参数是那种
		类型来决定使用那种方法获取第一个对象
		*/
		switch (q[0].charAt(0)) 
		{
			case ".":
			tem = q[0].slice(1);
			this.getClassName(tem);
			break;
			case "#":
			tem = q[0].slice(1);
			this.getId(tem);
			break;
			default:
			tem = q[0];
			this.getElement(tem);
			break;
		}

        /*数据由query修改传给findchild，
        用于计数有多少个参数传入
        */
        this.child = q.length 

		/*
		不断查找子元素，第一个参数由上
		面的方法得到了所以i从1开始
		*/
		for (var i = 1; i < q.length; i++) 
		{
			this.findchild(q[i])
		}
	}





	/*****************导入js文件*****************/
	/*
	append是在选定元素的子元素后面追加
	innerHTML导入的js代码不能执行要使用
    createElement
    使用：$().include("XXX.js")
    */
    this.include = function(x) 
    {
    	var body = document.getElementsByTagName("html")[0];
    	var content = document.createElement("srcipt");

    	content.setAttribute("src", x)
    	body.append(content)
    }





    /*****************颜色设置函数*****************/
	/*

	$("div").color("background:color, color:black")
	*/
	this.css = function(y) 
	{     
		var x=y.replace(" ","")
		var y = y.split(",")

		if (x == null) 
		{
			alert("css函数传入的不是标准值")
			return;
		}

		for (var i = 0; i < y.length; i++) 
		{
			if (y[i] != "") 
			{
				var u = y[i].split(":")

				for (var j = 0; j < this.element.length; j++) 
				{
					this.element[j].style[u[0]] = u[1] //style的点号可以使用[]替代，这样就能动态设置了。
				}
			}
		}
		return this
	}






	/*****************得到元素*****************/
	/*
	使用：$("XX"||空值).getEl(数值) 
	得到元素后就可以使用js原生的方法了。
	*/
	this.getEl=function()
	{
		var ar=arguments;
		var len=ar.length;
		var _temp
		if(this.element.length==0)
		{
			alert("你还没有选定元素");
			return;
		}

		if(len!=0)
		{  
			if(arguments[0]=="all")
			{  
				if(arguments[1]!=null)
				{
					alert("getEl('all'"+",'"+arguments[1]+"')\n不能有后面的参数:"+arguments[1])
				}
				else
				{
					console.log("getEl('all')\n返回的结果只能使用原生方法")
				}
				return this.element
			}
			if(this.element[arguments[0]]==null)
			{
				alert("元素集中没有下表为："+arguments[0]+"\n的元素")
				this.element=[]
				return  this;
			}
			if(arguments[1]=="$")
			{
				_temp=this.element[arguments[0]]
				this.element=[]
				this.element.push(_temp)
				return this
			}
			else
			{   

				if(arguments[1]!=null)
				{
					alert("getEl("+arguments[0]+",'"+arguments[1]+"')\n不能有后面的参数:"+arguments[1])
				}
				else
				{
					console.log("getEl("+arguments[0]+")\n返回的结果只能使用原生方法")
				}
				return this.element[arguments[0]]
			}
		}

	}



	

	/*****************3D效果*****************/
	/*
	j:加载的图片数量
	src:图片的位置
	id：你自己定义的容器id
	shengdu:表示精深
	z:图片到中心的距离
	#_wrap img:设置图片颜色
	#_wrap:设置包裹的div的样式
	使用如：$(id).D3(2,"file/img",100,100)
	*/
	this.D3=function(j,src,shengdu,z)
	{
		/*
		先添加包装元素
		*/	
		var loc=this.element[0]
		if(loc==null)
		{
			alert("还没有选定嵌入的位置")
			return
		}
		var insertwrap="<div id='_wrap'> </div>"
		loc.innerHTML=insertwrap

		var wrap=document.getElementById("_wrap")
		wrap.style.transformStyle="preserve-3d"
		wrap.style.transform="rotateX(-10deg)"
		wrap.style.position="relative"

		var insertstack=[]
		var insertimg=""

		//设置景深
		loc.style.perspective=shengdu+"px"
		for(var k=1;k<j+1;k++ )
		{   
			insertstack.push("<img src='"+src+"/"+k+".jpg'>")

		}
		/*
		如果直接在for中使用length
		由于每次弹栈length都会改变
		经验就是：不在for()中计算
		*/
		var len=insertstack.length
		for(var i=0;i<len;i++)
		{  
			insertimg+=insertstack.shift()
		}

		wrap.innerHTML=insertimg


       	/*
       	设置仰视图和俯视图时
       	的取阴影效果 
       	*/
       	var tr=function()
       	{

       		var dg=owrap.style.transform.toString()
       		var r=/rotateX\([\d]+/ig
       		var dg=r.exec(dg)
       		var st
       		for(var i=0;i<len;i++)
       		{
       			oimg[i].style.webkitBoxReflect="below 5px -webkit-linear-gradient(top,rgba(0,0,0,0) 40%,rgba(0,0,0,0.5) 100%)" 
       		}
       		if(dg!=null)
       		{   
       			st=dg.toString().split("(")
       			st=parseInt(st[1])

       			if(st>1)
       			{  
       				
       				var idg=wrap.style.transform.toString()
       				var ir=/rotateY\(-*[\d]+/ig
       				var idg=ir.exec(idg)


       				if(idg!=null)
       				{	

       					var ist=idg.toString().split("(")
       					var ist=parseInt(ist[1])%360

       					var ty=0;
       					if(ist>j)
       					{
							var ding=parseInt(ist/deg)//走了多少步
							for(var uo=parseInt(j/4);uo>-parseInt(j/4);uo--)    //移位
							{ 
								if(uo-ding<0) //开头要在每个数上轮一次才叫遍历了
								          	 //当最后的一个开始为-1时还要向后走，由于开头的还没到0的位置
								{            //所以还要向前走。但是这是-1是表示最大数了，这时要+个数，就变成了最大的下标
									ty=j
								}

                             	if(uo-ding<-j) //已经超出一个数量级了移位导致还超出一个数量级所以还要加一个数量级
                             	{
                             		ty=2*j
                             	}
								oimg[uo-ding+ty].style.webkitBoxReflect=null  //9-dind退了多少步

							}
						}
						if(ist<0)
						{  
							var ding=parseInt(ist/deg)//走了多少步

							for(var uo=parseInt(j/4);uo>-parseInt(j/4);uo--)    //移位
							{   
								if(uo-ding<0) //开头要在每个数上轮一次才叫遍历了
								          			//当最后的一个开始为-1时还要向后走，由于开头的还没到0的位置
								{              //所以还要向前走。但是这是-1是表示最大数了，这时要+个数，就变成了最大的下标
									ty=j
								}

                             	if(uo-ding>j-1) //已经超出一个数量级了移位导致还超出一个数量级所以还要加一个数量级
                             	{
                             		ty=-j
                             		oimg[uo-ding+ty].style.webkitBoxReflect=null
                             		ty=0      //for中第一个为uo-ding为j,ty=-j，但是第二个不为j，但ty还是为-j导致
                             	}             //else中的oimg错误。所以要恢复
                             	else
                             	{
									oimg[uo-ding+ty].style.webkitBoxReflect=null  //9-dind退了多少步
								}
							}
						}

					}

				}

			}
		}







		/************* 
		以下是3D核心代码*
		****************/
		var oimg = document.getElementsByTagName("img");
		var owrap = document.getElementById("_wrap")
		var deg = 360 / (oimg.length);
		var len = oimg.length;
		var timer;

		owrap.style.position="relative"
		
		
		for (var i=0; i < len; i++)
		{   
			oimg[i].style.position="absolute"
			oimg[i].style.webkitBoxReflect="below 5px -webkit-linear-gradient(top,rgba(0,0,0,0) 40%,rgba(0,0,0,0.5) 100%)" 
			oimg[i].style.transform = "rotateY(" + i * deg + "deg)  translateZ("+z+"px) ";
			oimg[i].style.zindex = -i*2;
			oimg[i].style.WebkitTransition = "all 2s";
			oimg[i].style.transition = "all "+(i+1)+"s";
			oimg[i].style.top="0px"
			

		}



		var nowx = 0,
		nowy = 0,
		minusx = 0,
		minusy = 0,
		rotx = 0,
		roty = 0;
		document.onmousedown = function(e)
		{
			e.preventDefault();
			var e = e || window.event;
			lastx = e.clientX, lasty = e.clientY,
			this.onmousemove = function(e) {
				tr()
				var e = e || window.event;
				nowx = e.clientX;
				nowy = e.clientY;

				minusx = nowx - lastx;
				minusy = nowy - lasty;

				rotx -= minusy;
				roty += minusx;

				if (rotx < -30) {
					rotx = -30
				}
				if (rotx > 30) {
					rotx = 30
				}

				// transform会清空所有的设置。所以
				// 一起设置，否则最后一次有效。
				owrap.style.transform = "rotateX(" + rotx + "deg)  rotateY(" + roty + "deg)";

				lastx = nowx;
				lasty = nowy;
				return this
			}

			this.onmouseup = function(e)
			{
				/*清除事件*/
				this.onmousemove=null

				/*惯性*/
				timer = setInterval(function() 
				{
					minusx *= 0.98;
					minusy *= 0.98;
					rotx -= minusy * 0.1
					roty += minusx * 0.1

					if (rotx < -30) 
					{
						rotx = -30
					}
					if (rotx > 30) 
					{
						rotx = 30
					}
					
					owrap.style.transform = "rotateX(" + rotx + "deg) rotateY(" + roty + "deg)";
					tr()
					if (Math.abs(minusx) < 0.1 || Math.abs(minusy) < 0.1) 
					{
						clearInterval(timer)
					}
				}, 1000 / 60)
			}
		}
	}






	/*****************在开始元素之前插入*****************/
	/*
	newnode:表示要插入的对象
	只能是包裹着html标签的文本否则
	多次添加导致冲掉先前添加的内容。
	使用$(xx).befroeStart(newnode)
	返回 this便于连缀
	*/
	this.beforeStart=function(newnode)
	{   
		var insertnode
		var len=this.element.length
		for(var i=0;i<len;i++)
		{   
			insertnode=this.element[i]
			if(this.element[i]=="undefined"||this.element[i]==null)
			{
				alert("beforeStart中传入了一个空值而中断")
				return	
			}

			insertnode=this.element[i]
			this.insertBefore(newnode,insertnode);

		}
		return this
		
	}







	/*****************在元素开始节点后面添加*****************/
	/*
	newel:表示要添加的内容
	只能是包裹着html标签的文本否则
	多次添加导致冲掉先前添加的内容。
	返回this便于连缀
	使用方法：$(xxxx).afterStart(newel)
	*/
	this.afterStart=function(newel)
	{   
		var el
		var len=this.element.length
		var child
		var content

		for(var i=0;i<len;i++)
		{   
			el=this.element[i]
			if(el=="undefined")
			{
				alert("afterStart中传入空值而中断")
				return
			}
			child=el.children
			content=el.innerHTML
			if(typeof(child[0])!="undefined")
			{   
				this.insertBefore(newel,child[0])
			}
			else 
			{   
				el.innerHTML=""
				el.append(newel)
				el.append(content)
			}
		}
		return this
	}




	/*****************在结束元素之前添加*****************/
	/*
	newel:要添加的内容
	只能是包裹着html标签的文本否则
	多次添加导致冲掉先前添加的内容。
	返回this便于连缀
	使用方法：$(XXX).beforeEnd(newel) 
	*/
	this.beforeEnd=function(newel)
	{   
		var el
		var len=this.element.length
		for(var i=0;i<len;i++ )
		{   
			el=this.element[i]
			if(el=="undefined")
			{
				alert("beforeEnd中传入空值而中断")
				return
			}
			el.innerHTML=el.innerHTML+newel
		}
		return this
	}







	/*****************在结束元素之后插入*****************/
	/*
	insertnode:表示要插入点元素
	newnode：表示插入的内容
	只能是包裹着html标签的文本否则
	多次添加导致冲掉先前添加的内容。
	使用方法$(XX).afterEnd(newnode)
	返回值this便于连缀
	*/
	this.afterEnd=function(newnode)
	{   
		var len=this.element.length
		var insertnode
		var nextbrother
		for(var i=0;i<len;i++)
		{   
			insertnode=this.element[i]
			if(insertnode=="undefined")
			{
				alert("afterEnd传入一个空值而中断")
				return 
			}
			
			if(insertnode.nodeType!=1)
			{
				alert("传入的对象集中有不是元素节点的afterEnd")
				return ;
			}

			/*
			知己写的用于查找兄弟的nextbrotherling有bug
			会把换行符也算进去 
			*/
			nextbrother=this.brother(insertnode,"next")

			if(nextbrother==null)
			{   
				insertnode.parentNode.append(newnode)
			}
			else if(nextbrother!=null)
			{   
				this.insertBefore(newnode,nextbrother)
			}
		}
		return this
	}






	/*****************空格分割工具*****************/
	/*
	x:表示字符串 
	注意大写会变成小写
	返回list
	*/
	this.tool = function(x)
	{
		var h = x.toLowerCase().split(" ");
		var tol = [];

		for (var i = 0; i < h.length; i++)
		{
			if (h[i] != "") 
			{
				tol.push(h[i])
			}
		}
		return tol
	}







	/******************获取函数名称***************/

	/*使用了正则表达式,
	"/要匹配的样式/ig" i表示忽略大小写，g是global
	表示全局.

	匿名函数（使用变量来代表这个函数。本身没有函数名）
	变量代表的是赋的值，不代表本身
	要获取本身叫啥使用.name获取
	*/
	this.getFunc = function(str) 
	{

		var e = /^function\s+[A-z]+/ig //得到 function  x（）
		var t = /\s[A-z]+/ig //得到空格+x()
		var y = /[A-z]+/ig //去除空格
		var op = e.exec(str)
		var ko = t.exec(op)

		ko = y.exec(ko)
		ko = ko.toString()
        //ko本来为null但是使用了tostring变成了"null"
        if(ko=="null")
        	ko=str.name

        return ko
    }






    /********************插件扩展******************/
	/*
	使用方法是：$().extend(XXX)
	XXX是函数的名称，不要括号。 
	xxx一般不能是匿名函数（var ji=fucntion(){AA}
   ji是变量代表这个匿名的函数不是函数名称，
   变量不是其字面名称是所代表的值）
   但是在上面getFunc（X）函数中进行了调优
   ，使用x.name把变量的名称获取出来--匿名函数名
   为null我使用变量名来命名插件的方法
   。
	
   */
   this.extend = function(x) 
   {   
		var name = this.getFunc(x) //获取函数的名称
		/*prototype 不能使用this 只能使用类的 
		名称使用[]号替代点号可以动态设置名称.
		由于$函数中每次是new一个对象，所以使用
		prototype 设置插件，这样左右的对象可以
		共享这个插件
		*/
		chajian.prototype[name] = x
		return this
	}







	/*****************查找同级别的元素*****************/
	/*
	el：表示要查找的元素
	flag：有“next”表示查找下一个同级别的元素
	有“pre” 表示查找上一个级别的同级元素  
	*/
	this.brother=function (el,flag)
	{
		var parent=el.parentNode
		var children=parent.children
		var len=children.length
		var  stack=[]
		for(var i=0;i<len;i++)
		{  
			stack.push(children[i])
		}

		var index=stack.indexOf(el)

		if(flag=="next")
		{
			index+=1
		}
		if(flag=="pre")
		{
			index-=1

		}
		if(stack[index]==null)
		{
			return null
		}
		return stack[index]
	}








	/*****************重写的原生的insertBefore*****************/
	/*
	content：要添加的内容
	注意:
	只要求输入一个值，
	如果单独使用
	可以输入两个值。
	第一个参数是要添加的内容
	第二个是元素对象

	没有返回值
	使用1方法：$(XXX).insertBefore(content)
	使用2方法：$().insertBefore(content,元素)
    content不能是纯文本(带有html标签)，否则多
    次添加有可能冲掉前面的添加内容
    */
    this.insertBefore=function(content)
    {   
    	var temp=""
    	var parent=""
    	var parentchildren=""
    	var tempchildren=[]
    	var index=""
    	var len=""
    	var ellen=""
    	var contact=""

    	if(arguments[1]=="undefined")
    	{   
    		temp=this.element
    		ellen=temp.length
    		for(var i=0;i<ellen;i++)
    		{
    			if(temp[i]=="undefined")
    			{
    				alert("insertBefore中传入一个空值而中断")
    				return 
    			}

    			parent=temp[i].parentNode
    			parentchildren=parent.children
    			len=parentchildren.length 


    			for(var n=0;n<len;n++)
    			{
    				tempchildren.push(parentchildren[n])
    			}

    			index=tempchildren.indexOf(temp[i])

    			for(var i=0;i<len;i++)
    			{   
    				if(i==index)
    				{
    					contact+=content
    				}

    				var attr=parentchildren[i].attributes
    				var alen=attr.length
    				var acon=" "
    				for(var i=0;i<alen;i++)
    				{
    					acon+=attr[i].name+"="+attr[i].value+" "
    				}
    				contact+="<"+parentchildren[i].tagName.toLowerCase()+
    				acon+
    				">"+parentchildren[i].innerHTML+
    				"</"+parentchildren[i].tagName.toLowerCase()+">"
    			}
    		}
    	}
    	if(arguments[1]!="undefined"){
    		temp=arguments[1]

    		if(temp=="undefined")
    		{
    			alert("insertBefore中传入一个空值而中断")
    			return 
    		}


    		parent=temp.parentNode
    		parentchildren=parent.children
    		len=parentchildren.length 



    		for(var n=0;n<len;n++)
    		{
    			tempchildren.push(parentchildren[n])
    		}

    		index=tempchildren.indexOf(temp)

    		for(var k=0;k<len;k++)
    		{   
    			if(k==index)
    			{
    				contact+=content
    			}
    			var attr=parentchildren[k].attributes
    			var alen=attr.length
    			var acon=" "
    			for(var i=0;i<alen;i++)
    			{   
    				acon+=attr[i].name+"="+attr[i].value+" "
    			}


    			contact+="<"+parentchildren[k].tagName.toLowerCase()+
    			acon+
    			">"+parentchildren[k].innerHTML+
    			"</"+parentchildren[k].tagName.toLowerCase()+">"
    		}
    	}
    	if(parent!="undefined")
    	{
    		parent.innerHTML=contact
			/*
			innerHMTL是要等所有程序完成后
			一次性渲染，所以两个innerHTML
			中间有其他程序时不会先渲染第一个
			而是到最后一个innerHTML时才渲染；
			但是再次定位元素就会使其先渲染
			后面的再次渲染
			*/
			this.query(_x) 
		}
	}

}








