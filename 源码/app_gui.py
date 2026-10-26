import copy,json,os,queue,subprocess,sys,time,uuid,shutil
from pathlib import Path
import tkinter as tk
from tkinter import ttk,messagebox,filedialog,simpledialog
from app_paths import APP_NAME,BASE,data_dir,read_json,write_json
from app_settings import defaults,load_profile,save_profile,validate,STRATEGIES,SAMPLE_RULES
from app_catalog import validate_overrides
from app_windows import windows,find_target,activate,StopEvent,Job,driver_installed

PLUGIN_TEMPLATE='''# plan(context, state) 每次选路时调用。返回规则动作；返回 {} 使用内置策略。
# context: length, elapsed, nearest_ball, near_wall, head_cell, eye, path_trust
# state 是原始识别状态。请避免在这里读文件、截图、休眠或执行耗时工作。
def plan(context, state):
    if context['length'] >= 40 and context['elapsed'] >= 45:
        return {'prefer_survival': True, 'disable_food': ['grn']}
    return {}
'''

class ScrollPage(ttk.Frame):
    def __init__(self,parent):
        super().__init__(parent)
        self.canvas=tk.Canvas(self,highlightthickness=0)
        scroll=ttk.Scrollbar(self,orient='vertical',command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scroll.set);scroll.pack(side='right',fill='y');self.canvas.pack(side='left',fill='both',expand=True)
        self.body=ttk.Frame(self.canvas,padding=16);item=self.canvas.create_window((0,0),window=self.body,anchor='nw')
        self.body.bind('<Configure>',lambda _:self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.bind('<Configure>',lambda e:self.canvas.itemconfigure(item,width=e.width))
        def wheel(event):
            if self.winfo_ismapped()and self.canvas.winfo_rootx()<=event.x_root<self.canvas.winfo_rootx()+self.canvas.winfo_width()and self.canvas.winfo_rooty()<=event.y_root<self.canvas.winfo_rooty()+self.canvas.winfo_height():self.canvas.yview_scroll(int(-event.delta/120),'units')
        self.bind_all('<MouseWheel>',wheel,add='+')

class App:
    def __init__(self,smoke_path=None,acceptance=False,acceptance_matches=1):
        self.root=tk.Tk();self.root.title(APP_NAME)
        height=min(780,max(610,self.root.winfo_screenheight()-100))
        self.root.geometry('1040x%d'%height);self.root.minsize(880,610)
        try:self.root.iconbitmap(str(BASE/'app.ico'))
        except tk.TclError:pass
        style=ttk.Style();style.theme_use('clam')
        style.configure('.',font=('Microsoft YaHei UI',10),padding=3)
        style.configure('Title.TLabel',font=('Microsoft YaHei UI',20,'bold'),foreground='#174c45')
        style.configure('Accent.TButton',font=('Microsoft YaHei UI',12,'bold'),padding=(22,12))
        style.configure('Hint.TLabel',foreground='#596663')
        style.configure('Treeview',rowheight=28,font=('Microsoft YaHei UI',9))
        try:self.profile=load_profile()
        except Exception:self.profile=defaults()
        self.catalog=read_json(BASE/'parameter_catalog.json');self.by_id={p['id']:p for p in self.catalog['parameters']}
        self.vars={};self.events=queue.Queue();self.process=None;self.job=None;self.stop_event=None
        self.run_dir=None;self.stopping_at=0;self.exiting=False;self.acceptance=acceptance;self.window_rows=[]
        from app_tray import Tray
        self.tray=Tray(self.events.put)
        self._build();self._load_fields();self.refresh_windows();self.filter_parameters()
        from app_notice import NoticeWatchdog
        self.notice_watchdog=NoticeWatchdog();self.notice_reported=False
        from app_paths import data_notice
        if data_notice:self.detail.set(data_notice)
        try:
            import mss
            with mss.mss()as sct:
                if len(sct.monitors)>2:
                    m=next((m for m in sct.monitors[1:]if m.get('is_primary')is False),sct.monitors[-1])
                    height=min(780,max(610,m['height']-120));width=min(1040,max(880,m['width']-80))
                    self.root.geometry('%dx%d%+d%+d'%(width,height,m['left']+40,m['top']+30))
        except Exception:pass
        self.root.after(100,self.show)
        self.root.protocol('WM_DELETE_WINDOW',self.on_close)
        self.root.after(250,self.poll)
        if smoke_path:self.root.after(1200,lambda:self.smoke(smoke_path))
        if acceptance:
            self.var('run','matches').set(str(acceptance_matches));self.var('run','infinite',True).set(False)
            self.root.after(2000,self.start)

    def run(self):self.root.mainloop()
    def var(self,section,key,boolean=False):
        ident=section+'.'+key
        if ident not in self.vars:self.vars[ident]=tk.BooleanVar()if boolean else tk.StringVar()
        return self.vars[ident]
    def entry(self,parent,label,section,key,row,width=12):
        ttk.Label(parent,text=label).grid(row=row,column=0,sticky='w',padx=10,pady=5)
        ttk.Entry(parent,textvariable=self.var(section,key),width=width).grid(row=row,column=1,sticky='w',padx=10,pady=5)
    def check(self,parent,label,section,key,**grid):
        ttk.Checkbutton(parent,text=label,variable=self.var(section,key,True)).grid(sticky='w',padx=10,pady=5,**grid)
    def _build(self):
        top=ttk.Frame(self.root,padding=(20,14));top.pack(fill='x')
        ttk.Label(top,text='蛇对蛇助手',style='Title.TLabel').pack(side='left')
        ttk.Label(top,text='2.5 · 作者 101010 · 仅供学习交流',style='Hint.TLabel').pack(side='left',padx=16,pady=(10,0))
        ttk.Button(top,text='打开记录文件夹',command=self.open_records).pack(side='right')
        ttk.Button(top,text='使用声明',command=lambda:self.tabs.select(self.notice_page)).pack(side='right',padx=6)
        self.tabs=ttk.Notebook(self.root);self.tabs.pack(fill='both',expand=True,padx=20)
        homepage=ScrollPage(self.tabs);basicpage=ScrollPage(self.tabs);strategypage=ScrollPage(self.tabs)
        screenpage=ScrollPage(self.tabs);home=homepage.body;screen=screenpage.body
        basic=basicpage.body;strategy=strategypage.body
        advanced=ttk.Frame(self.tabs,padding=12);rules=ttk.Frame(self.tabs,padding=12)
        for frame,name in((homepage,'运行'),(screenpage,'画面适配'),(basicpage,'控制与规划'),(strategypage,'策略开关'),(advanced,'全部参数'),(rules,'编写策略')):self.tabs.add(frame,text=name)
        self.notice_page=ttk.Frame(self.tabs,padding=12);self.tabs.add(self.notice_page,text='声明与署名')
        from app_notice import verify_distribution
        notice=verify_distribution(check_assets=False)
        from tkinter.scrolledtext import ScrolledText
        reader=ScrolledText(self.notice_page,wrap='word',font=('Microsoft YaHei UI',10),padx=12,pady=10)
        reader.insert('1.0',notice);reader.configure(state='disabled');reader.pack(fill='both',expand=True)
        runbox=ttk.LabelFrame(home,text='自动游戏',padding=16);runbox.pack(fill='x',pady=(4,16))
        self.entry(runbox,'自动玩几局','run','matches',0)
        self.check(runbox,'持续运行，直到手动停止','run','infinite',row=1,column=0,columnspan=2)
        self.entry(runbox,'每局之间等待（秒）','run','pause_seconds',2)
        self.entry(runbox,'启动准备时间（秒）','run','start_delay',3)
        self.check(runbox,'游戏失去前台时暂停控制','run','pause_on_focus_loss',row=4,column=0,columnspan=2)
        controls=ttk.Frame(home);controls.pack(fill='x',pady=4)
        self.start_button=ttk.Button(controls,text='开始运行',style='Accent.TButton',command=self.start);self.start_button.pack(side='left')
        self.stop_button=ttk.Button(controls,text='停止',command=self.stop,state='disabled');self.stop_button.pack(side='left',padx=12)
        ttk.Label(home,text='开始后窗口自动隐藏到右下角托盘。停止快捷键：Ctrl + Alt + F8',style='Hint.TLabel').pack(anchor='w',pady=(20,8))
        ttk.Label(home,text='让游戏停在匹配大厅，并保持游戏画面可见。识别到大厅后自动按 X；结算后自动返回。',wraplength=860,style='Hint.TLabel').pack(anchor='w')
        self.status=tk.StringVar(value='就绪');self.detail=tk.StringVar(value='默认沿用2.5：长按A，START后45秒松开；复活后重新按住。')
        ttk.Label(home,textvariable=self.status,font=('Microsoft YaHei UI',14,'bold')).pack(anchor='w',pady=(26,6))
        ttk.Label(home,textvariable=self.detail,wraplength=860).pack(anchor='w')
        ttk.Label(screen,text='自动按游戏窗口大小换算，支持1280×720至3840×2160及窗口模式。',wraplength=860).pack(anchor='w',pady=(0,12))
        row=ttk.Frame(screen);row.pack(fill='x',pady=6)
        ttk.Label(row,text='游戏窗口').pack(side='left');self.window_combo=ttk.Combobox(row,state='readonly',width=60);self.window_combo.pack(side='left',padx=10)
        ttk.Button(row,text='刷新',command=self.refresh_windows).pack(side='left')
        self.window_combo.bind('<<ComboboxSelected>>',self.select_window)
        form=ttk.Frame(screen);form.pack(anchor='w',pady=12)
        ttk.Label(form,text='截图方式').grid(row=0,column=0,sticky='w',padx=10,pady=6)
        ttk.Combobox(form,state='readonly',values=['自动游戏窗口','整个显示器','手动棋盘区域'],textvariable=self.var('capture','mode'),width=20).grid(row=0,column=1,sticky='w',padx=10)
        self.entry(form,'显示器编号（从1开始）','capture','monitor',1)
        self.check(form,'按16:9内容区域适配（带黑边时保留）','capture','fit_16_9',row=2,column=0,columnspan=2)
        self.check(form,'看到“匹配”但没有X时，按菜单键唤起手柄模式','run','controller_wakeup',row=3,column=0,columnspan=2)
        self.entry(form,'缺少X的确认时长（秒）','run','controller_confirm_seconds',4)
        self.entry(form,'菜单键重试间隔（秒）','run','controller_retry_seconds',5)
        self.entry(form,'每局最多尝试菜单键次数','run','controller_max_attempts',6)
        self.entry(form,'“匹配”文字识别阈值','capture','lobby_word_threshold',7)
        self.entry(form,'X图标识别阈值','capture','lobby_x_threshold',8)
        buttons=ttk.Frame(screen);buttons.pack(anchor='w',pady=10)
        for text,command in(('预览识别区域',self.preview),('框选游戏内容',lambda:self.calibrate('viewport')),('框选棋盘',lambda:self.calibrate('board')),('恢复自动区域',self.reset_capture)):
            ttk.Button(buttons,text=text,command=command).pack(side='left',padx=(0,8))
        self.capture_text=tk.StringVar();ttk.Label(screen,textvariable=self.capture_text,wraplength=850,style='Hint.TLabel').pack(anchor='w',pady=16)
        ttk.Label(screen,text='预览中的棋盘应完整覆盖29×25格。若游戏有额外黑边，先框选完整游戏内容；若HUD比例不同，再框选棋盘。大厅提示会随游戏内容区域一起适配。',wraplength=850,style='Hint.TLabel').pack(anchor='w',pady=8)
        left=ttk.LabelFrame(basic,text='手柄与响应',padding=8);left.pack(side='left',fill='both',expand=True,padx=(0,10))
        right=ttk.LabelFrame(basic,text='长蛇与运行',padding=8);right.pack(side='left',fill='both',expand=True)
        ttk.Label(left,text='按键加速方式').grid(row=0,column=0,sticky='w',padx=10)
        ttk.Combobox(left,state='readonly',values=['持续按A','不按A','策略自行选择'],textvariable=self.var('controls','accel_mode'),width=16).grid(row=0,column=1,padx=10,pady=6)
        for row,(label,key) in enumerate([('松开A时间（START后秒）','release_seconds'),('转向提前量（毫秒）','turn_lead_ms'),('复活松A时长（秒）','respawn_release_seconds'),('图像计算线程','cv_threads'),('截图线程（1或2）','capture_threads')],1):self.entry(left,label,'controls',key,row)
        for row,(label,key) in enumerate([('进入长蛇策略的长度','long_threshold'),('退出长蛇策略的长度','long_exit_threshold'),('局末保长时间（秒）','long_protect_at'),('窄口计算预算（毫秒）','topology_budget_ms'),('逃生路线有效期（秒）','escape_hint_seconds')]):self.entry(right,label,'strategy_options',key,row)
        self.entry(right,'开始后最长运行（秒）','run','after_start_seconds',5)
        self.entry(right,'每隔几帧保存（0关闭）','run','save_frames',6)
        self.entry(right,'等待超时（分钟，0不限）','run','waiting_timeout_minutes',7)
        ttk.Label(strategy,text='关闭单项策略会退回基础选路；追食开关只影响主动寻路，沿途仍可能吃到食物。',wraplength=850,style='Hint.TLabel').grid(row=0,column=0,columnspan=2,sticky='w',pady=(0,16))
        for i,(key,label) in enumerate(STRATEGIES.items()):self.check(strategy,label,'strategies',key,row=1+i//2,column=i%2)
        ttk.Label(strategy,text='规则与插件在“编写策略”页编辑，保存后于下一次启动生效。',style='Hint.TLabel').grid(row=8,column=0,columnspan=2,sticky='w',pady=20)
        toolbar=ttk.Frame(advanced);toolbar.pack(fill='x')
        self.search=tk.StringVar();self.category=tk.StringVar(value='全部')
        ttk.Label(toolbar,text='搜索').pack(side='left');entry=ttk.Entry(toolbar,textvariable=self.search,width=32);entry.pack(side='left',padx=6)
        self.search.trace_add('write',lambda *_:self.filter_parameters())
        combo=ttk.Combobox(toolbar,textvariable=self.category,state='readonly',width=18,values=['全部']+sorted({p['category']for p in self.catalog['parameters']}));combo.pack(side='left',padx=6);combo.bind('<<ComboboxSelected>>',lambda _:self.filter_parameters())
        ttk.Button(toolbar,text='仅看已修改',command=lambda:self.search.set('@已修改')).pack(side='left')
        self.param_count=tk.StringVar();ttk.Label(advanced,textvariable=self.param_count,style='Hint.TLabel').pack(anchor='w',pady=6)
        area=ttk.Frame(advanced);area.pack(fill='both',expand=True)
        self.tree=ttk.Treeview(area,columns=('module','name','value'),show='headings',selectmode='browse')
        for key,text,width in(('module','模块 / 分类',170),('name','参数 / 所在函数',440),('value','当前值',150)):
            self.tree.heading(key,text=text);self.tree.column(key,width=width,minwidth=60)
        scroll=ttk.Scrollbar(area,orient='vertical',command=self.tree.yview);self.tree.configure(yscrollcommand=scroll.set);self.tree.pack(side='left',fill='both',expand=True);scroll.pack(side='right',fill='y')
        self.tree.bind('<<TreeviewSelect>>',self.parameter_selected);self.tree.bind('<Double-1>',lambda _:self.edit_parameter())
        self.param_detail=tk.StringVar(value='参数覆盖包括运行中的配置常量和算法数值。固定棋盘规格由画面适配页管理。')
        ttk.Label(advanced,textvariable=self.param_detail,wraplength=910,style='Hint.TLabel').pack(anchor='w',pady=8)
        pbuttons=ttk.Frame(advanced);pbuttons.pack(fill='x')
        ttk.Button(pbuttons,text='修改所选参数',command=self.edit_parameter).pack(side='left')
        ttk.Button(pbuttons,text='恢复所选',command=self.reset_parameter).pack(side='left',padx=8)
        ttk.Label(pbuttons,text='高级参数改动可能影响识别和路径安全，请保留可回退的配置。',style='Hint.TLabel').pack(side='left',padx=8)
        sub=ttk.Notebook(rules);sub.pack(fill='both',expand=True)
        rulepage=ttk.Frame(sub);pluginpage=ttk.Frame(sub);sub.add(rulepage,text='JSON规则');sub.add(pluginpage,text='Python插件')
        ttk.Label(rulepage,text='按顺序匹配条件，后面的动作可覆盖前面的动作。保存后需在策略开关页启用。',wraplength=860,style='Hint.TLabel').pack(anchor='w',pady=8)
        self.rule_editor=tk.Text(rulepage,font=('Consolas',10),wrap='none',undo=True);self.rule_editor.pack(fill='both',expand=True)
        ttk.Button(rulepage,text='插入示例规则',command=self.sample_rules).pack(anchor='w',pady=8)
        pathrow=ttk.Frame(pluginpage);pathrow.pack(fill='x',pady=8)
        ttk.Entry(pathrow,textvariable=self.var('root','plugin_path'),width=65).pack(side='left',fill='x',expand=True)
        ttk.Button(pathrow,text='打开插件',command=self.choose_plugin).pack(side='left',padx=6)
        ttk.Button(pathrow,text='保存编辑内容',command=self.save_plugin).pack(side='left')
        ttk.Label(pluginpage,text='本地插件与软件拥有相同权限，仅加载你自己编写或信任的文件。单次执行超过20毫秒或出错时停用并回退。',wraplength=870,style='Hint.TLabel').pack(anchor='w',pady=6)
        self.plugin_editor=tk.Text(pluginpage,font=('Consolas',10),wrap='none',undo=True);self.plugin_editor.pack(fill='both',expand=True)
        self.plugin_editor.insert('1.0',PLUGIN_TEMPLATE)
        ttk.Button(pluginpage,text='新建示例插件',command=self.new_plugin).pack(anchor='w',pady=8)
        footer=ttk.Frame(self.root,padding=(20,10));footer.pack(fill='x',side='bottom',before=self.tabs)
        for text,command in(('保存设置',self.save),('导出配置',self.export_profile),('导入配置',self.import_profile),('恢复2.5默认',self.reset_all)):
            ttk.Button(footer,text=text,command=command).pack(side='left',padx=(0,8))
        ttk.Label(footer,text='设置在下次开始时生效',style='Hint.TLabel').pack(side='right')

    def _load_fields(self):
        for ident,var in self.vars.items():
            section,key=ident.split('.',1);value=self.profile.get(key,'')if section=='root'else self.profile[section][key]
            if ident=='capture.mode':value={'window':'自动游戏窗口','monitor':'整个显示器','manual':'手动棋盘区域'}[value]
            if ident=='controls.accel_mode':value={'hold':'持续按A','off':'不按A','adaptive':'策略自行选择'}[value]
            var.set(value)
        self.rule_editor.delete('1.0','end');self.rule_editor.insert('1.0',json.dumps(self.profile['rules'],ensure_ascii=False,indent=2))
        self.update_capture_text()
        path=Path(self.profile.get('plugin_path')or '__not_selected__')
        if path.is_file():
            try:self.plugin_editor.delete('1.0','end');self.plugin_editor.insert('1.0',path.read_text(encoding='utf-8-sig'))
            except OSError:pass

    def collect(self):
        p=copy.deepcopy(self.profile)
        for ident,var in self.vars.items():
            section,key=ident.split('.',1);value=var.get()
            if section=='root':p[key]=value;continue
            old=defaults()[section][key]
            if ident=='capture.mode':value={'自动游戏窗口':'window','整个显示器':'monitor','手动棋盘区域':'manual'}[value]
            elif ident=='controls.accel_mode':value={'持续按A':'hold','不按A':'off','策略自行选择':'adaptive'}[value]
            elif isinstance(old,int)and not isinstance(old,bool):
                try:value=int(value)
                except ValueError:raise ValueError(key+'需要填写整数。')
            elif isinstance(old,float):
                try:value=float(value)
                except ValueError:raise ValueError(key+'需要填写数值。')
            p[section][key]=value
        p['rules']=json.loads(self.rule_editor.get('1.0','end'))
        p=validate(p);validate_overrides(p,self.catalog);return p

    def save(self,notify=True):
        try:self.profile=save_profile(self.collect());self.filter_parameters()
        except PermissionError:
            messagebox.showerror('配置目录无法写入','当前配置目录没有写入权限。\n'+str(data_dir())+'\n请确认该目录未被安全软件拦截；原设置尚未覆盖，游戏控制没有启动。',parent=self.root);return False
        except Exception as exc:
            messagebox.showerror('设置未保存',str(exc),parent=self.root);return False
        if notify:messagebox.showinfo('设置已保存','将在下次开始时生效。',parent=self.root)
        return True
    def export_profile(self):
        if not self.save(False):return
        path=filedialog.asksaveasfilename(title='导出配置',defaultextension='.json',initialfile='蛇对蛇2.5配置.json',filetypes=[('配置文件','*.json')])
        if path:write_json(path,self.profile)
    def import_profile(self):
        path=filedialog.askopenfilename(title='导入配置',filetypes=[('配置文件','*.json')])
        if not path:return
        try:
            p=validate(read_json(path,{}));validate_overrides(p,self.catalog);self.profile=p;self._load_fields();self.filter_parameters();self.refresh_windows()
        except Exception as exc:messagebox.showerror('无法导入',str(exc),parent=self.root)
    def reset_all(self):
        if not messagebox.askyesno('恢复默认','将当前界面所有设置恢复为2.5默认值？',parent=self.root):return
        self.profile=defaults();self._load_fields();self.refresh_windows();self.filter_parameters()
    def refresh_windows(self):
        self.window_rows=windows();labels=['%s  [%s×%s]'%(r['title'],r['rect'][2],r['rect'][3])for r in self.window_rows]
        self.window_combo.configure(values=labels)
        expected=self.profile['capture']['target_title']
        index=next((i for i,r in enumerate(self.window_rows)if r['title']==expected),None)
        if index is not None:self.window_combo.current(index);self.select_window()
        else:self.window_combo.set('未找到游戏，打开后点击刷新')
    def select_window(self,_=None):
        index=self.window_combo.current()
        if 0<=index<len(self.window_rows):
            row=self.window_rows[index];self.profile['capture'].update(target_title=row['title'],target_hwnd=row['hwnd'])
    def update_capture_text(self):
        capture=self.profile['capture'];self.capture_text.set('内容区域：%s\n棋盘区域：%s'%('自定义'if capture['viewport']else'自动适配','自定义'if capture['manual_board']else'自动适配'))
    def reset_capture(self):
        self.profile['capture'].update(manual_board=None,viewport=None);self.var('capture','mode').set('自动游戏窗口');self.update_capture_text()

    def preview(self):
        self.root.withdraw();activate(find_target(self.profile['capture']))
        self.root.after(500,self._preview_capture)
    def _preview_capture(self):
        try:
            from app_capture import Locator,AdaptiveVision,AdaptiveLobby
            import cv2
            from PIL import Image,ImageTk
            locator=Locator(self.collect()['capture']);vision=AdaptiveVision(locator);lobby=AdaptiveLobby(locator)
            try:image=vision.grab();evidence=lobby.observe();box=vision.last_box
            finally:vision.close();lobby.close()
            if box is None:raise ValueError('未找到所选游戏窗口。')
            image=cv2.resize(image,(696,600));photo=ImageTk.PhotoImage(Image.fromarray(image))
            self.show();window=tk.Toplevel(self.root);window.title('棋盘识别区域');label=ttk.Label(window,image=photo);label.image=photo;label.pack()
            ttk.Label(window,text='棋盘位置：%s\n“匹配”文字：%.3f（%s）　X图标：%.3f（%s）'%(box,evidence['word_score'],'已识别'if evidence['word_visible']else'未识别',evidence['x_score'],'已识别'if evidence['x_visible']else'未识别'),wraplength=690).pack(padx=8,pady=8)
            ttk.Button(window,text='关闭',command=window.destroy).pack(pady=6)
        except Exception as exc:self.show();messagebox.showerror('无法预览',str(exc),parent=self.root)

    def calibrate(self,kind):
        self.root.withdraw();activate(find_target(self.profile['capture']))
        self.root.after(500,lambda:self._calibrate_capture(kind))
    def _calibrate_capture(self,kind):
        try:
            import mss
            from PIL import Image,ImageTk
            from app_windows import client_rect
            rect=client_rect(find_target(self.profile['capture']))
            if not rect:raise ValueError('请先选择已打开的游戏窗口。')
            x,y,w,h=rect
            with mss.mss()as sct:
                raw=sct.grab(dict(left=x,top=y,width=w,height=h));image=Image.frombytes('RGB',raw.size,raw.rgb)
            scale=min(1.,1000/w,620/h);shown=image.resize((round(w*scale),round(h*scale)))
            self.show();window=tk.Toplevel(self.root);window.title('拖动框选'+('完整游戏内容'if kind=='viewport'else'完整棋盘')+'，按Esc取消')
            photo=ImageTk.PhotoImage(shown);canvas=tk.Canvas(window,width=shown.width,height=shown.height,cursor='crosshair');canvas.pack();canvas.create_image(0,0,image=photo,anchor='nw');canvas.photo=photo
            selection={'start':None,'id':None}
            def down(event):
                selection['start']=(event.x,event.y)
                if selection['id']:canvas.delete(selection['id'])
                selection['id']=canvas.create_rectangle(event.x,event.y,event.x,event.y,outline='#ff5353',width=3)
            def drag(event):
                if selection['start']:canvas.coords(selection['id'],*selection['start'],event.x,event.y)
            def up(event):
                if not selection['start']:return
                a,b=selection['start'];l,r=sorted((a,event.x));t,bot=sorted((b,event.y))
                l=max(0,l);t=max(0,t);r=min(shown.width,r);bot=min(shown.height,bot)
                if r-l<40 or bot-t<40:return
                if kind=='viewport':self.profile['capture']['viewport']=[l/shown.width,t/shown.height,(r-l)/shown.width,(bot-t)/shown.height]
                else:
                    self.profile['capture']['manual_board']=[l/shown.width,t/shown.height,r/shown.width,bot/shown.height];self.var('capture','mode').set('手动棋盘区域')
                self.update_capture_text();window.destroy()
            canvas.bind('<Button-1>',down);canvas.bind('<B1-Motion>',drag);canvas.bind('<ButtonRelease-1>',up);window.bind('<Escape>',lambda _:window.destroy())
        except Exception as exc:self.show();messagebox.showerror('无法框选',str(exc),parent=self.root)

    def current_value(self,item):
        if item['kind']=='config':return self.profile['config_overrides'].get(item['name'],item['default'])
        return self.profile['literal_overrides'].get(item['id'],item['default'])
    def changed(self,item):
        return item['name']in self.profile['config_overrides']if item['kind']=='config'else item['id']in self.profile['literal_overrides']
    def filter_parameters(self):
        if not hasattr(self,'tree'):return
        query=self.search.get().lower().strip();category=self.category.get()
        items=[]
        for p in self.catalog['parameters']:
            if category!='全部'and p['category']!=category:continue
            haystack=' '.join(str(p[k])for k in('id','category','function','context','default')).lower()
            if (self.changed(p)if query=='@已修改'else all(word in haystack for word in query.split())):items.append(p)
        self.tree.delete(*self.tree.get_children())
        for item in items[:600]:
            value=json.dumps(self.current_value(item),ensure_ascii=False)
            self.tree.insert('','end',iid=item['id'],values=(item['category'],('★ 'if self.changed(item)else'')+item['name'],value[:100]))
        self.param_count.set('共%d项；当前匹配%d项，显示前%d项。双击修改；按模块、函数或源码表达式搜索。'%(len(self.by_id),len(items),min(600,len(items))))
    def parameter_selected(self,_=None):
        chosen=self.tree.selection()
        if not chosen:return
        p=self.by_id[chosen[0]];self.param_detail.set('%s · %s · 默认值 %s\n%s'%(p['id'],p['type'],json.dumps(p['default'],ensure_ascii=False)[:100],p['context'][:210]))
    def edit_parameter(self):
        chosen=self.tree.selection()
        if not chosen:return
        p=self.by_id[chosen[0]]
        if not p['editable']:messagebox.showinfo('由画面适配管理','此项是固定棋盘或输出位置，请使用画面适配设置。',parent=self.root);return
        text=simpledialog.askstring('修改参数',p['id']+'\n'+p['context'][:250]+'\n按JSON格式填写；布尔值用true/false。',initialvalue=json.dumps(self.current_value(p),ensure_ascii=False),parent=self.root)
        if text is None:return
        try:
            value=json.loads(text);trial=copy.deepcopy(self.profile);key='config_overrides'if p['kind']=='config'else'literal_overrides';ident=p['name']if p['kind']=='config'else p['id'];trial[key][ident]=value;validate_overrides(trial,self.catalog);self.profile=trial;self.filter_parameters()
        except Exception as exc:messagebox.showerror('参数无效',str(exc),parent=self.root)
    def reset_parameter(self):
        chosen=self.tree.selection()
        if not chosen:return
        p=self.by_id[chosen[0]];key='config_overrides'if p['kind']=='config'else'literal_overrides';ident=p['name']if p['kind']=='config'else p['id'];self.profile[key].pop(ident,None);self.filter_parameters()
    def sample_rules(self):
        self.rule_editor.delete('1.0','end');self.rule_editor.insert('1.0',json.dumps(SAMPLE_RULES,ensure_ascii=False,indent=2))
    def new_plugin(self):
        self.plugin_editor.delete('1.0','end');self.plugin_editor.insert('1.0',PLUGIN_TEMPLATE);self.var('root','plugin_path').set('')
    def choose_plugin(self):
        path=filedialog.askopenfilename(title='选择本地策略插件',filetypes=[('Python插件','*.py')])
        if path:
            try:self.plugin_editor.delete('1.0','end');self.plugin_editor.insert('1.0',Path(path).read_text(encoding='utf-8-sig'));self.var('root','plugin_path').set(path)
            except Exception as exc:messagebox.showerror('无法读取',str(exc),parent=self.root)
    def save_plugin(self):
        path=self.var('root','plugin_path').get()
        if not path:path=filedialog.asksaveasfilename(title='保存插件',initialdir=str(data_dir()),initialfile='my_strategy.py',defaultextension='.py',filetypes=[('Python插件','*.py')])
        if not path:return
        try:
            source=self.plugin_editor.get('1.0','end');compile(source,path,'exec');Path(path).write_text(source,encoding='utf8');self.var('root','plugin_path').set(path)
            messagebox.showinfo('插件已保存','可在策略开关页启用。',parent=self.root)
        except Exception as exc:messagebox.showerror('无法保存',str(exc),parent=self.root)

    def start(self):
        if self.process and self.process.poll()is None:return
        from app_notice import verify_distribution,NoticeIntegrityError
        try:verify_distribution()
        except NoticeIntegrityError as exc:
            messagebox.showerror('使用声明校验失败',str(exc)+'\n请恢复完整原始软件包。',parent=self.root);return
        if not self.save(False):return
        if not driver_installed():messagebox.showerror('需要虚拟手柄驱动','请先安装ViGEmBus驱动。下载地址见使用说明。',parent=self.root);return
        hwnd=find_target(self.profile['capture'])
        if not hwnd:messagebox.showerror('未找到游戏','请打开游戏，在画面适配页选择游戏窗口。',parent=self.root);return
        self.profile['capture']['target_hwnd']=hwnd
        self.run_dir=data_dir()/'records'/time.strftime('%Y%m%d_%H%M%S')
        self.run_dir=self.run_dir.with_name(self.run_dir.name+'_'+uuid.uuid4().hex[:4]);self.run_dir.mkdir(parents=True)
        worker_profile=copy.deepcopy(self.profile)
        if worker_profile['strategies']['plugin_enabled']:
            plugin=self.run_dir/'strategy_plugin.py';shutil.copy2(worker_profile['plugin_path'],plugin);worker_profile['plugin_path']=str(plugin)
        write_json(self.run_dir/'profile.json',worker_profile)
        from app_paths import APP_VERSION
        write_json(self.run_dir/'version.json',{'app':APP_VERSION,'baseline':'2.5 / round188','engine_id':self.catalog['engine_id']})
        name='Local\\Snake25Stop_'+uuid.uuid4().hex
        self.stop_event=StopEvent(name);self.job=Job()
        command=([sys.executable]if getattr(sys,'frozen',False)else[sys.executable,str(BASE/'app.py')])+['--worker','--profile',str(self.run_dir/'profile.json'),'--event',name,'--status',str(self.run_dir/'status.json'),'--run-dir',str(self.run_dir)]
        try:
            self.process=subprocess.Popen(command,cwd=str(data_dir()),creationflags=subprocess.CREATE_NO_WINDOW)
            self.job.assign(self.process)
        except Exception as exc:
            if self.process and self.process.poll()is None:self.process.terminate()
            self.cleanup_worker();messagebox.showerror('无法开始',str(exc),parent=self.root);return
        self.stopping_at=0;self.start_button.configure(state='disabled');self.stop_button.configure(state='normal');self.status.set('正在准备');self.detail.set('记录保存至：'+str(self.run_dir))
        if self.tray.failed:self.root.iconify()
        else:self.root.withdraw()
        activate(hwnd)
        if self.acceptance:write_json(self.run_dir/'acceptance_ui.json',{'hidden':self.root.state()=='withdrawn','tray_ready':self.tray.failed is None,'started':time.time()})
    def stop(self):
        if self.stop_event and self.process and self.process.poll()is None:
            self.stop_event.set()
            if not self.stopping_at:self.stopping_at=time.monotonic()
            self.status.set('正在停止并释放手柄');self.tray.set_status('正在停止')
    def show(self):
        self.root.deiconify();self.root.lift();self.root.focus_force()
    def on_close(self):
        if self.process and self.process.poll()is None:
            if not self.tray.failed:self.root.withdraw()
            else:self.root.iconify()
        else:self.quit()
    def quit(self):
        self.exiting=True
        if self.process and self.process.poll()is None:self.stop();return
        self.cleanup_worker();self.notice_watchdog.close();self.tray.close();self.root.destroy()
    def cleanup_worker(self):
        if self.job:self.job.close();self.job=None
        if self.stop_event:self.stop_event.close();self.stop_event=None
    def poll(self):
        if self.notice_watchdog.error and not self.notice_reported:
            self.notice_reported=True;self.stop();self.start_button.configure(state='disabled')
            self.status.set('使用声明校验失败，已停止控制');self.show()
            messagebox.showerror('使用声明校验失败',self.notice_watchdog.error+'\n请恢复完整原始软件包后重新启动。',parent=self.root)
        try:
            while True:
                action=self.events.get_nowait()
                if action=='show':self.show()
                elif action=='stop':self.stop()
                elif action=='quit':
                    self.quit()
                    if not self.process:return
        except queue.Empty:pass
        if not self.root.winfo_exists():return
        if self.process:
            status=read_json(self.run_dir/'status.json',{})
            if status:
                total='不限'if self.profile['run']['infinite']else str(status.get('total')or self.profile['run']['matches'])
                text='%s · 第%s/%s局 · 已完成%s局'%(status.get('phase','运行中'),status.get('current',0),total,status.get('completed',0))
                self.status.set(text);self.tray.set_status(text)
                self.detail.set('%s\n本局计时 %.1f秒 · 估计长度 %s（以游戏结算为准）\n%s'%(status.get('message',''),status.get('elapsed',0),status.get('length',0),self.run_dir))
            if self.stopping_at and time.monotonic()-self.stopping_at>8 and self.process.poll()is None:
                self.job.close();self.job=None;self.status.set('已强制结束异常的控制进程')
            if self.process.poll()is not None:
                code=self.process.returncode;self.process=None;self.cleanup_worker()
                self.start_button.configure(state='disabled'if self.notice_reported else'normal');self.stop_button.configure(state='disabled')
                state=status.get('state');message=status.get('message')or('运行结束'if code==0 else'运行失败，请查看记录中的console.log')
                self.status.set(message);self.tray.set_status(message);self.show()
                if self.acceptance:
                    write_json(self.run_dir/'acceptance_ui_done.json',{'exit_code':code,'state':state,'completed':status.get('completed',0),'shown':self.root.state()=='normal'})
                if self.exiting or self.acceptance:self.quit();return
        self.root.after(250,self.poll)
    def open_records(self):
        path=self.run_dir or data_dir()/'records';path.mkdir(parents=True,exist_ok=True);os.startfile(str(path))
    def smoke(self,path):
        from PIL import ImageGrab
        self.root.update();x=self.root.winfo_rootx();y=self.root.winfo_rooty();w=self.root.winfo_width();h=self.root.winfo_height()
        ImageGrab.grab(bbox=(x,y,x+w,y+h),all_screens=True).save(path);self.quit()
