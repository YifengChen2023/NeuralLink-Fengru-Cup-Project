function varargout = untitled1(varargin)
% UNTITLED1 MATLAB code for untitled1.fig
%      UNTITLED1, by itself, creates a new UNTITLED1 or raises the existing
%      singleton*.
%
%      H = UNTITLED1+ returns the handle to a new UNTITLED1 or the handle to
%      the existing singleton*.
%
%      UNTITLED1('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in UNTITLED1.M with the given input arguments.
%
%      UNTITLED1('Property','Value',...) creates a new UNTITLED1 or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before untitled1_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to untitled1_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help untitled1

% Last Modified by GUIDE v2.5 11-May-2023 01:01:28

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @untitled1_OpeningFcn, ...
                   'gui_OutputFcn',  @untitled1_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% --- Executes just before untitled1 is made visible.
function untitled1_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to untitled (see VARARGIN)

% Choose default command line output for untitled
handles.output = hObject;

% clear plot
% cla(handles.AF3);cla(handles.F3);
% cla(handles.F7);cla(handles.FC5);
% cla(handles.T7);cla(handles.P7);
% cla(handles.O1);
% cla(handles.AF4);cla(handles.F4);
% cla(handles.F8);cla(handles.FC6);
% cla(handles.T8);cla(handles.P8);
% cla(handles.O2);

% tpc连接
global tcpclient;
tcpclient = tcpip('127.0.0.1', 8000, 'Timeout', 60,'OutputBufferSize',10240,'InputBufferSize',10240);%连接这个ip和这个端口的UDP服务器
fopen(tcpclient);
fwrite(tcpclient,'please sent');%发送一段数据给tcp服务器。服务器好知道matlab的ip和端口

% 删除现有的定时器，重新创建定时器
delete(timerfind); 
h1=timer;   %定时器
handles.h1=h1;   %将定时器放到全局变量中
set(handles.h1,'ExecutionMode','fixedRate');   %定时器，循环执行，循环定时。
set(handles.h1,'Period',0.1);    %定时器
set(handles.h1,'TimerFcn',{@dispplot,handles});
start(handles.h1);

handles.output = hObject;

% background
I = imread('exit.png');   %读取图片
I = imresize(I, [50, 100]);
set(handles.exit,'CData',I);  %将按钮的背景图片设置成I，美化按钮
I = imread('clear.png');   %读取图片
I = imresize(I, [60, 100]);
set(handles.clear,'CData',I);  %将按钮的背景图片设置成I，美化按钮
I = imread('onoff.png');   %读取图片
I = imresize(I, [70, 200]);
set(handles.start_end,'CData',I);  %将按钮的背景图片设置成I，美化按钮

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes untitled1 wait for user response (see UIRESUME)
% uiwait(handles.figure1);
global DATA;
DATA = zeros(14, 200);
MEAN = [4309.7886574891, 4298.7697202038335, 4330.342540854722, 4326.737623612304,...
        4290.042517634397, 4290.283844391647, 4294.674507826712, 4305.180294637339, ...
        4282.67872789123, 4271.779960641943, 4486.425850332993, 4421.121165935583, ...
        4384.4559363871585, 4445.615973379307]; 
for i = 1:14
    DATA(i, :) = MEAN(i);
end
global data;
data = zeros(15, 2);
global onoff;
onoff = 0;
clear_Callback(hObject, eventdata, handles)
global mod;
mod = 1;
function dispplot(hObject, eventdata, handles)
set(handles.time,'String',['当前时间为：',datestr(now, 31)]);   % 将edit控件的内容改成当前时间
global onoff;
global DATA;
global tcpclient;
global data;
global mod;
len = 200;
fwrite(tcpclient,'please sent');%发送一段数据给tcp服务器。服务器好知道matlab的ip和端口
nBytes = get(tcpclient,'BytesAvailable');
if nBytes>0
    receive = fread(tcpclient,nBytes);
    data_t = str2num(char(receive(2:end-1)'));
    data(:, 1) = data_t(1:15);
    data(:, 2) = data_t(16:30);
    rate = double(data_t(1, 31) * 100);
    rate = rate / 1280;
    if rate > 60
        rate = round((rate - 100) / 40 * 100);
    else
        rate = round(rate / 60 * 100);
    end
    set(handles.process,'String',[num2str(rate), '%']);
%     data(:, 3) = data_t(31:45);
%     data(:, 4) = data_t(46:30);
end

if onoff
    width = 1.2;
    x = (1:1:len);
    gap = 2;
    if ~isnan(data)
        DATA(1:14, 1:198) = DATA(1:14, 3:200);
        DATA(1:14, 199: 200) = data(1:14, 1:2);
    end
    
    plot(handles.AF3, x, DATA(1, :), 'r', 'Linewidth', width);
    xlim(handles.AF3,[0, len]);
    plot(handles.F3, x,  DATA(2, :), 'color','#1E90FF', 'Linewidth', width);
    xlim(handles.F3,[0, len]);
    plot(handles.F7, x,  DATA(3, :), 'color', '#FFD700', 'Linewidth', width);
    xlim(handles.F7,[0, len]);
    plot(handles.FC5, x,  DATA(4, :), 'color','#6B8E23', 'Linewidth', width);
    xlim(handles.FC5,[0, len]);
    plot(handles.T7, x,  DATA(5, :), 'color', '#800080', 'Linewidth', width);
    xlim(handles.T7,[0, len]);
    plot(handles.P7, x,  DATA(6, :), 'color', '#0000CD', 'Linewidth', width);
    xlim(handles.P7,[0, len]);
    plot(handles.O1, x, DATA(7, :), 'color', '#FFA500', 'Linewidth', width);
    xlim(handles.O1,[0, len]);
    plot(handles.AF4, x, DATA(8, :), 'r', 'Linewidth', width);
    xlim(handles.AF4,[0, len]);
    plot(handles.F4, x,  DATA(9, :),  'color','#1E90FF', 'Linewidth', width);
    xlim(handles.F4,[0, len]);
    plot(handles.F8, x,  DATA(10, :), 'color', '#FFD700', 'Linewidth', width);
    xlim(handles.F8,[0, len]);
    plot(handles.FC6, x,  DATA(11, :), 'color','#6B8E23', 'Linewidth', width);
    xlim(handles.FC6,[0, len]);
    plot(handles.T8, x,  DATA(12, :), 'color', '#800080', 'Linewidth', width);
    xlim(handles.T8,[0, len]);
    plot(handles.P8, x,  DATA(13, :), 'color', '#0000CD', 'Linewidth', width);
    xlim(handles.P8,[0, len]);
    plot(handles.O2, x, DATA(14, :),  'color', '#FFA500', 'Linewidth', width);
    xlim(handles.O2,[0, len]);
    
    if(data(15, 1) == 0)
        set(handles.command,'string','等待指令');
    elseif(data(15, 1) == 1)
        if(mod == 1)
            set(handles.command,'string','当前指令为：前翻');
        elseif(mod == 2)
            set(handles.command,'string','当前指令为：向前');
        else
            set(handles.command,'string','当前指令为：一字');
        end
    elseif(data(15, 1) == 2)
        if(mod == 1)
            set(handles.command,'string','当前指令为：左翻');
        elseif(mod == 2)
            set(handles.command,'string','当前指令为：向左');
        else
            set(handles.command,'string','当前指令为：三角');
        end            
    elseif(data(15, 1) == 3)
        if(mod == 1)
            set(handles.command,'string','当前指令为：右翻');
        elseif(mod == 2)
            set(handles.command,'string','当前指令为：向右');
        else
            set(handles.command,'string','当前指令为：旋转');
        end
    elseif(data(15, 1) == 4)
        if(mod == 1)
            set(handles.command,'string','当前指令为：后翻');
        elseif(mod == 2)
            set(handles.command,'string','当前指令为：向后');
        else
            set(handles.command,'string','当前指令为：阶梯');
        end
    else
        set(handles.command,'string','空');
    end
end

% --- Outputs from this function are returned to the command line.
function varargout = untitled1_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% ha=axes('units','normalized','pos',[0 0 1 1]);
% uistack(ha,'bottom');    %置于底部用buttom
% ii=imread('bg.png');   %在当前文件夹下的图片名称
% image(ii);
% colormap gray
% set(ha,'handlevisibility','off','visible','off');


% --- Executes on button press in start_end.
function start_end_Callback(hObject, eventdata, handles)
global onoff;
if onoff == 0
   onoff = 1;
   set(handles.command,'string','当前指令为：空');
else
   onoff = 0;
   set(handles.command,'string','当前已暂停');
end
global tcpclient;
fwrite(tcpclient,['switch=', num2str(onoff), 'ok']);


% --- Executes on button press in exit.
function exit_Callback(hObject, eventdata, handles)
stop(handles.h1);
delete(handles.h1);
global tcpclient;
delete(tcpclient);
close()


% --- Executes on button press in clear.
function clear_Callback(hObject, eventdata, handles)
global DATA;
global data;
MEAN = [4309.7886574891, 4298.7697202038335, 4330.342540854722, 4326.737623612304,...
        4290.042517634397, 4290.283844391647, 4294.674507826712, 4305.180294637339, ...
        4282.67872789123, 4271.779960641943, 4486.425850332993, 4421.121165935583, ...
        4384.4559363871585, 4445.615973379307];
for i = 1:14
    DATA(i, :) = MEAN(i);
end
len = 200;
data = zeros(15, 2);
x = (1:1:len);
width = 1.2;
plot(handles.AF3, x, DATA(1, :), 'r', 'Linewidth', width);
xlim(handles.AF3,[0, len]);
plot(handles.F3, x,  DATA(2, :), 'color','#1E90FF', 'Linewidth', width);
xlim(handles.F3,[0, len]);
plot(handles.F7, x,  DATA(3, :), 'color', '#FFD700', 'Linewidth', width);
xlim(handles.F7,[0, len]);
plot(handles.FC5, x,  DATA(4, :), 'color','#6B8E23', 'Linewidth', width);
xlim(handles.FC5,[0, len]);
plot(handles.T7, x,  DATA(5, :), 'color', '#800080', 'Linewidth', width);
xlim(handles.T7,[0, len]);
plot(handles.P7, x,  DATA(6, :), 'color', '#0000CD', 'Linewidth', width);
xlim(handles.P7,[0, len]);
plot(handles.O1, x, DATA(7, :), 'color', '#FFA500', 'Linewidth', width);
plot(handles.AF4, x, DATA(8, :), 'r', 'Linewidth', width);
xlim(handles.AF4,[0, len]);
plot(handles.F4, x,  DATA(9, :),  'color','#1E90FF', 'Linewidth', width);
xlim(handles.F4,[0, len]);
plot(handles.F8, x,  DATA(10, :), 'color', '#FFD700', 'Linewidth', width);
xlim(handles.F8,[0, len]);
plot(handles.FC6, x,  DATA(11, :), 'color','#6B8E23', 'Linewidth', width);
xlim(handles.FC6,[0, len]);
plot(handles.T8, x,  DATA(12, :), 'color', '#800080', 'Linewidth', width);
xlim(handles.T8,[0, len]);
plot(handles.P8, x,  DATA(13, :), 'color', '#0000CD', 'Linewidth', width);
xlim(handles.P8,[0, len]);
plot(handles.O2, x, DATA(14, :),  'color', '#FFA500', 'Linewidth', width);
xlim(handles.O2,[0, len]);


% --- Executes on selection change in mod.
function mod_Callback(hObject, eventdata, handles)
% hObject    handle to mod (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns mod contents as cell array
%        contents{get(hObject,'Value')} returns selected item from mod
global mod;
global tcpclient;
mod = get(handles.mod,'value');
fwrite(tcpclient,['mode=', num2str(mod), 'ok']);

% --- Executes during object creation, after setting all properties.
function mod_CreateFcn(hObject, eventdata, handles)
% hObject    handle to mod (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
