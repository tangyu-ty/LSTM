Ѓќ6
хє
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(ѕ
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕе
@
StaticRegexFullMatch	
input

output
"
patternstring
Ш
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ф
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handleіжelement_dtype"
element_dtypetype"

shape_typetype:
2	
џ
TensorListReserve
element_shape"
shape_type
num_elements#
handleіжelement_dtype"
element_dtypetype"

shape_typetype:
2	
ѕ
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint         
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ
ћ
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
ѕ"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28█б3
z
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_19/kernel
s
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel*
_output_shapes

:*
dtype0
r
dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_19/bias
k
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
_output_shapes
:*
dtype0
z
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_20/kernel
s
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel*
_output_shapes

:*
dtype0
r
dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_20/bias
k
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
_output_shapes
:*
dtype0
z
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_21/kernel
s
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel*
_output_shapes

:*
dtype0
r
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_21/bias
k
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes
:*
dtype0
z
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_22/kernel
s
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel*
_output_shapes

:*
dtype0
r
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_22/bias
k
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
њ
lstm_21/lstm_cell_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*,
shared_namelstm_21/lstm_cell_21/kernel
І
/lstm_21/lstm_cell_21/kernel/Read/ReadVariableOpReadVariableOplstm_21/lstm_cell_21/kernel*
_output_shapes

:@*
dtype0
д
%lstm_21/lstm_cell_21/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*6
shared_name'%lstm_21/lstm_cell_21/recurrent_kernel
Ъ
9lstm_21/lstm_cell_21/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_21/lstm_cell_21/recurrent_kernel*
_output_shapes

:@*
dtype0
і
lstm_21/lstm_cell_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namelstm_21/lstm_cell_21/bias
Ѓ
-lstm_21/lstm_cell_21/bias/Read/ReadVariableOpReadVariableOplstm_21/lstm_cell_21/bias*
_output_shapes
:@*
dtype0
њ
lstm_22/lstm_cell_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*,
shared_namelstm_22/lstm_cell_22/kernel
І
/lstm_22/lstm_cell_22/kernel/Read/ReadVariableOpReadVariableOplstm_22/lstm_cell_22/kernel*
_output_shapes

:@*
dtype0
д
%lstm_22/lstm_cell_22/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*6
shared_name'%lstm_22/lstm_cell_22/recurrent_kernel
Ъ
9lstm_22/lstm_cell_22/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_22/lstm_cell_22/recurrent_kernel*
_output_shapes

:@*
dtype0
і
lstm_22/lstm_cell_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namelstm_22/lstm_cell_22/bias
Ѓ
-lstm_22/lstm_cell_22/bias/Read/ReadVariableOpReadVariableOplstm_22/lstm_cell_22/bias*
_output_shapes
:@*
dtype0
њ
lstm_23/lstm_cell_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*,
shared_namelstm_23/lstm_cell_23/kernel
І
/lstm_23/lstm_cell_23/kernel/Read/ReadVariableOpReadVariableOplstm_23/lstm_cell_23/kernel*
_output_shapes

:@*
dtype0
д
%lstm_23/lstm_cell_23/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*6
shared_name'%lstm_23/lstm_cell_23/recurrent_kernel
Ъ
9lstm_23/lstm_cell_23/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_23/lstm_cell_23/recurrent_kernel*
_output_shapes

:@*
dtype0
і
lstm_23/lstm_cell_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namelstm_23/lstm_cell_23/bias
Ѓ
-lstm_23/lstm_cell_23/bias/Read/ReadVariableOpReadVariableOplstm_23/lstm_cell_23/bias*
_output_shapes
:@*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
ѕ
Adam/dense_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_19/kernel/m
Ђ
*Adam/dense_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/m*
_output_shapes

:*
dtype0
ђ
Adam/dense_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_19/bias/m
y
(Adam/dense_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_19/bias/m*
_output_shapes
:*
dtype0
ѕ
Adam/dense_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_20/kernel/m
Ђ
*Adam/dense_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/m*
_output_shapes

:*
dtype0
ђ
Adam/dense_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_20/bias/m
y
(Adam/dense_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/m*
_output_shapes
:*
dtype0
ѕ
Adam/dense_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_21/kernel/m
Ђ
*Adam/dense_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/m*
_output_shapes

:*
dtype0
ђ
Adam/dense_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_21/bias/m
y
(Adam/dense_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/m*
_output_shapes
:*
dtype0
ѕ
Adam/dense_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_22/kernel/m
Ђ
*Adam/dense_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/m*
_output_shapes

:*
dtype0
ђ
Adam/dense_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_22/bias/m
y
(Adam/dense_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/m*
_output_shapes
:*
dtype0
а
"Adam/lstm_21/lstm_cell_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*3
shared_name$"Adam/lstm_21/lstm_cell_21/kernel/m
Ў
6Adam/lstm_21/lstm_cell_21/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_21/lstm_cell_21/kernel/m*
_output_shapes

:@*
dtype0
┤
,Adam/lstm_21/lstm_cell_21/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*=
shared_name.,Adam/lstm_21/lstm_cell_21/recurrent_kernel/m
Г
@Adam/lstm_21/lstm_cell_21/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_21/lstm_cell_21/recurrent_kernel/m*
_output_shapes

:@*
dtype0
ў
 Adam/lstm_21/lstm_cell_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/lstm_21/lstm_cell_21/bias/m
Љ
4Adam/lstm_21/lstm_cell_21/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_21/lstm_cell_21/bias/m*
_output_shapes
:@*
dtype0
а
"Adam/lstm_22/lstm_cell_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*3
shared_name$"Adam/lstm_22/lstm_cell_22/kernel/m
Ў
6Adam/lstm_22/lstm_cell_22/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_22/lstm_cell_22/kernel/m*
_output_shapes

:@*
dtype0
┤
,Adam/lstm_22/lstm_cell_22/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*=
shared_name.,Adam/lstm_22/lstm_cell_22/recurrent_kernel/m
Г
@Adam/lstm_22/lstm_cell_22/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_22/lstm_cell_22/recurrent_kernel/m*
_output_shapes

:@*
dtype0
ў
 Adam/lstm_22/lstm_cell_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/lstm_22/lstm_cell_22/bias/m
Љ
4Adam/lstm_22/lstm_cell_22/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_22/lstm_cell_22/bias/m*
_output_shapes
:@*
dtype0
а
"Adam/lstm_23/lstm_cell_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*3
shared_name$"Adam/lstm_23/lstm_cell_23/kernel/m
Ў
6Adam/lstm_23/lstm_cell_23/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_23/lstm_cell_23/kernel/m*
_output_shapes

:@*
dtype0
┤
,Adam/lstm_23/lstm_cell_23/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*=
shared_name.,Adam/lstm_23/lstm_cell_23/recurrent_kernel/m
Г
@Adam/lstm_23/lstm_cell_23/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_23/lstm_cell_23/recurrent_kernel/m*
_output_shapes

:@*
dtype0
ў
 Adam/lstm_23/lstm_cell_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/lstm_23/lstm_cell_23/bias/m
Љ
4Adam/lstm_23/lstm_cell_23/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_23/lstm_cell_23/bias/m*
_output_shapes
:@*
dtype0
ѕ
Adam/dense_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_19/kernel/v
Ђ
*Adam/dense_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/v*
_output_shapes

:*
dtype0
ђ
Adam/dense_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_19/bias/v
y
(Adam/dense_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_19/bias/v*
_output_shapes
:*
dtype0
ѕ
Adam/dense_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_20/kernel/v
Ђ
*Adam/dense_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/v*
_output_shapes

:*
dtype0
ђ
Adam/dense_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_20/bias/v
y
(Adam/dense_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/v*
_output_shapes
:*
dtype0
ѕ
Adam/dense_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_21/kernel/v
Ђ
*Adam/dense_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/v*
_output_shapes

:*
dtype0
ђ
Adam/dense_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_21/bias/v
y
(Adam/dense_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/v*
_output_shapes
:*
dtype0
ѕ
Adam/dense_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_22/kernel/v
Ђ
*Adam/dense_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/v*
_output_shapes

:*
dtype0
ђ
Adam/dense_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_22/bias/v
y
(Adam/dense_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/v*
_output_shapes
:*
dtype0
а
"Adam/lstm_21/lstm_cell_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*3
shared_name$"Adam/lstm_21/lstm_cell_21/kernel/v
Ў
6Adam/lstm_21/lstm_cell_21/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_21/lstm_cell_21/kernel/v*
_output_shapes

:@*
dtype0
┤
,Adam/lstm_21/lstm_cell_21/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*=
shared_name.,Adam/lstm_21/lstm_cell_21/recurrent_kernel/v
Г
@Adam/lstm_21/lstm_cell_21/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_21/lstm_cell_21/recurrent_kernel/v*
_output_shapes

:@*
dtype0
ў
 Adam/lstm_21/lstm_cell_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/lstm_21/lstm_cell_21/bias/v
Љ
4Adam/lstm_21/lstm_cell_21/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_21/lstm_cell_21/bias/v*
_output_shapes
:@*
dtype0
а
"Adam/lstm_22/lstm_cell_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*3
shared_name$"Adam/lstm_22/lstm_cell_22/kernel/v
Ў
6Adam/lstm_22/lstm_cell_22/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_22/lstm_cell_22/kernel/v*
_output_shapes

:@*
dtype0
┤
,Adam/lstm_22/lstm_cell_22/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*=
shared_name.,Adam/lstm_22/lstm_cell_22/recurrent_kernel/v
Г
@Adam/lstm_22/lstm_cell_22/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_22/lstm_cell_22/recurrent_kernel/v*
_output_shapes

:@*
dtype0
ў
 Adam/lstm_22/lstm_cell_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/lstm_22/lstm_cell_22/bias/v
Љ
4Adam/lstm_22/lstm_cell_22/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_22/lstm_cell_22/bias/v*
_output_shapes
:@*
dtype0
а
"Adam/lstm_23/lstm_cell_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*3
shared_name$"Adam/lstm_23/lstm_cell_23/kernel/v
Ў
6Adam/lstm_23/lstm_cell_23/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_23/lstm_cell_23/kernel/v*
_output_shapes

:@*
dtype0
┤
,Adam/lstm_23/lstm_cell_23/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*=
shared_name.,Adam/lstm_23/lstm_cell_23/recurrent_kernel/v
Г
@Adam/lstm_23/lstm_cell_23/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_23/lstm_cell_23/recurrent_kernel/v*
_output_shapes

:@*
dtype0
ў
 Adam/lstm_23/lstm_cell_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/lstm_23/lstm_cell_23/bias/v
Љ
4Adam/lstm_23/lstm_cell_23/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_23/lstm_cell_23/bias/v*
_output_shapes
:@*
dtype0

NoOpNoOp
ъj
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*┘i
value¤iB╠i B┼i
М
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
l
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
l
cell

state_spec
 	variables
!trainable_variables
"regularization_losses
#	keras_api
R
$	variables
%trainable_variables
&regularization_losses
'	keras_api
l
(cell
)
state_spec
*	variables
+trainable_variables
,regularization_losses
-	keras_api
R
.	variables
/trainable_variables
0regularization_losses
1	keras_api
h

2kernel
3bias
4	variables
5trainable_variables
6regularization_losses
7	keras_api
R
8	variables
9trainable_variables
:regularization_losses
;	keras_api
h

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
R
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
h

Fkernel
Gbias
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
R
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
h

Pkernel
Qbias
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
ћ
Viter

Wbeta_1

Xbeta_2
	Ydecay
Zlearning_rate2mо3mО<mп=m┘Fm┌Gm█Pm▄QmП[mя\m▀]mЯ^mр_mР`mсamСbmтcmТ2vу3vУ<vж=vЖFvвGvВPvьQvЬ[v№\v­]vы^vЫ_vз`vЗavшbvШcvэ
~
[0
\1
]2
^3
_4
`5
a6
b7
c8
29
310
<11
=12
F13
G14
P15
Q16
~
[0
\1
]2
^3
_4
`5
a6
b7
c8
29
310
<11
=12
F13
G14
P15
Q16
 
Г
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
regularization_losses
 
ј
i
state_size

[kernel
\recurrent_kernel
]bias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
 

[0
\1
]2

[0
\1
]2
 
╣

nstates
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
Г
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
	variables
trainable_variables
regularization_losses
ј
y
state_size

^kernel
_recurrent_kernel
`bias
z	variables
{trainable_variables
|regularization_losses
}	keras_api
 

^0
_1
`2

^0
_1
`2
 
й

~states
non_trainable_variables
ђlayers
Ђmetrics
 ѓlayer_regularization_losses
Ѓlayer_metrics
 	variables
!trainable_variables
"regularization_losses
 
 
 
▓
ёnon_trainable_variables
Ёlayers
єmetrics
 Єlayer_regularization_losses
ѕlayer_metrics
$	variables
%trainable_variables
&regularization_losses
Њ
Ѕ
state_size

akernel
brecurrent_kernel
cbias
і	variables
Іtrainable_variables
їregularization_losses
Ї	keras_api
 

a0
b1
c2

a0
b1
c2
 
┐
јstates
Јnon_trainable_variables
љlayers
Љmetrics
 њlayer_regularization_losses
Њlayer_metrics
*	variables
+trainable_variables
,regularization_losses
 
 
 
▓
ћnon_trainable_variables
Ћlayers
ќmetrics
 Ќlayer_regularization_losses
ўlayer_metrics
.	variables
/trainable_variables
0regularization_losses
[Y
VARIABLE_VALUEdense_19/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_19/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

20
31

20
31
 
▓
Ўnon_trainable_variables
џlayers
Џmetrics
 юlayer_regularization_losses
Юlayer_metrics
4	variables
5trainable_variables
6regularization_losses
 
 
 
▓
ъnon_trainable_variables
Ъlayers
аmetrics
 Аlayer_regularization_losses
бlayer_metrics
8	variables
9trainable_variables
:regularization_losses
[Y
VARIABLE_VALUEdense_20/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_20/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

<0
=1

<0
=1
 
▓
Бnon_trainable_variables
цlayers
Цmetrics
 дlayer_regularization_losses
Дlayer_metrics
>	variables
?trainable_variables
@regularization_losses
 
 
 
▓
еnon_trainable_variables
Еlayers
фmetrics
 Фlayer_regularization_losses
гlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
[Y
VARIABLE_VALUEdense_21/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_21/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

F0
G1

F0
G1
 
▓
Гnon_trainable_variables
«layers
»metrics
 ░layer_regularization_losses
▒layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
 
 
 
▓
▓non_trainable_variables
│layers
┤metrics
 хlayer_regularization_losses
Хlayer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
[Y
VARIABLE_VALUEdense_22/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_22/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

P0
Q1

P0
Q1
 
▓
иnon_trainable_variables
Иlayers
╣metrics
 ║layer_regularization_losses
╗layer_metrics
R	variables
Strainable_variables
Tregularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUElstm_21/lstm_cell_21/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%lstm_21/lstm_cell_21/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_21/lstm_cell_21/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUElstm_22/lstm_cell_22/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%lstm_22/lstm_cell_22/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_22/lstm_cell_22/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUElstm_23/lstm_cell_23/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%lstm_23/lstm_cell_23/recurrent_kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_23/lstm_cell_23/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE
 
^
0
1
2
3
4
5
6
7
	8

9
10
11
12

╝0
й1
 
 
 

[0
\1
]2

[0
\1
]2
 
▓
Йnon_trainable_variables
┐layers
└metrics
 ┴layer_regularization_losses
┬layer_metrics
j	variables
ktrainable_variables
lregularization_losses
 
 

0
 
 
 
 
 
 
 
 
 

^0
_1
`2

^0
_1
`2
 
▓
├non_trainable_variables
─layers
┼metrics
 кlayer_regularization_losses
Кlayer_metrics
z	variables
{trainable_variables
|regularization_losses
 
 

0
 
 
 
 
 
 
 
 
 

a0
b1
c2

a0
b1
c2
 
х
╚non_trainable_variables
╔layers
╩metrics
 ╦layer_regularization_losses
╠layer_metrics
і	variables
Іtrainable_variables
їregularization_losses
 
 

(0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

═total

╬count
¤	variables
л	keras_api
I

Лtotal

мcount
М
_fn_kwargs
н	variables
Н	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

═0
╬1

¤	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

Л0
м1

н	variables
~|
VARIABLE_VALUEAdam/dense_19/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_19/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_20/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_20/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_21/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_21/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_22/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_22/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_21/lstm_cell_21/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ёѓ
VARIABLE_VALUE,Adam/lstm_21/lstm_cell_21/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_21/lstm_cell_21/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_22/lstm_cell_22/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ёѓ
VARIABLE_VALUE,Adam/lstm_22/lstm_cell_22/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_22/lstm_cell_22/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_23/lstm_cell_23/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ёѓ
VARIABLE_VALUE,Adam/lstm_23/lstm_cell_23/recurrent_kernel/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_23/lstm_cell_23/bias/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_19/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_19/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_20/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_20/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_21/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_21/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_22/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_22/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_21/lstm_cell_21/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ёѓ
VARIABLE_VALUE,Adam/lstm_21/lstm_cell_21/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_21/lstm_cell_21/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_22/lstm_cell_22/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ёѓ
VARIABLE_VALUE,Adam/lstm_22/lstm_cell_22/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_22/lstm_cell_22/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_23/lstm_cell_23/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ёѓ
VARIABLE_VALUE,Adam/lstm_23/lstm_cell_23/recurrent_kernel/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_23/lstm_cell_23/bias/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ѕ
serving_default_lstm_21_inputPlaceholder*+
_output_shapes
:         *
dtype0* 
shape:         
Щ
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_21_inputlstm_21/lstm_cell_21/kernel%lstm_21/lstm_cell_21/recurrent_kernellstm_21/lstm_cell_21/biaslstm_22/lstm_cell_22/kernel%lstm_22/lstm_cell_22/recurrent_kernellstm_22/lstm_cell_22/biaslstm_23/lstm_cell_23/kernel%lstm_23/lstm_cell_23/recurrent_kernellstm_23/lstm_cell_23/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *3
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *-
f(R&
$__inference_signature_wrapper_682557
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
┤
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_19/kernel/Read/ReadVariableOp!dense_19/bias/Read/ReadVariableOp#dense_20/kernel/Read/ReadVariableOp!dense_20/bias/Read/ReadVariableOp#dense_21/kernel/Read/ReadVariableOp!dense_21/bias/Read/ReadVariableOp#dense_22/kernel/Read/ReadVariableOp!dense_22/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_21/lstm_cell_21/kernel/Read/ReadVariableOp9lstm_21/lstm_cell_21/recurrent_kernel/Read/ReadVariableOp-lstm_21/lstm_cell_21/bias/Read/ReadVariableOp/lstm_22/lstm_cell_22/kernel/Read/ReadVariableOp9lstm_22/lstm_cell_22/recurrent_kernel/Read/ReadVariableOp-lstm_22/lstm_cell_22/bias/Read/ReadVariableOp/lstm_23/lstm_cell_23/kernel/Read/ReadVariableOp9lstm_23/lstm_cell_23/recurrent_kernel/Read/ReadVariableOp-lstm_23/lstm_cell_23/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_19/kernel/m/Read/ReadVariableOp(Adam/dense_19/bias/m/Read/ReadVariableOp*Adam/dense_20/kernel/m/Read/ReadVariableOp(Adam/dense_20/bias/m/Read/ReadVariableOp*Adam/dense_21/kernel/m/Read/ReadVariableOp(Adam/dense_21/bias/m/Read/ReadVariableOp*Adam/dense_22/kernel/m/Read/ReadVariableOp(Adam/dense_22/bias/m/Read/ReadVariableOp6Adam/lstm_21/lstm_cell_21/kernel/m/Read/ReadVariableOp@Adam/lstm_21/lstm_cell_21/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_21/lstm_cell_21/bias/m/Read/ReadVariableOp6Adam/lstm_22/lstm_cell_22/kernel/m/Read/ReadVariableOp@Adam/lstm_22/lstm_cell_22/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_22/lstm_cell_22/bias/m/Read/ReadVariableOp6Adam/lstm_23/lstm_cell_23/kernel/m/Read/ReadVariableOp@Adam/lstm_23/lstm_cell_23/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_23/lstm_cell_23/bias/m/Read/ReadVariableOp*Adam/dense_19/kernel/v/Read/ReadVariableOp(Adam/dense_19/bias/v/Read/ReadVariableOp*Adam/dense_20/kernel/v/Read/ReadVariableOp(Adam/dense_20/bias/v/Read/ReadVariableOp*Adam/dense_21/kernel/v/Read/ReadVariableOp(Adam/dense_21/bias/v/Read/ReadVariableOp*Adam/dense_22/kernel/v/Read/ReadVariableOp(Adam/dense_22/bias/v/Read/ReadVariableOp6Adam/lstm_21/lstm_cell_21/kernel/v/Read/ReadVariableOp@Adam/lstm_21/lstm_cell_21/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_21/lstm_cell_21/bias/v/Read/ReadVariableOp6Adam/lstm_22/lstm_cell_22/kernel/v/Read/ReadVariableOp@Adam/lstm_22/lstm_cell_22/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_22/lstm_cell_22/bias/v/Read/ReadVariableOp6Adam/lstm_23/lstm_cell_23/kernel/v/Read/ReadVariableOp@Adam/lstm_23/lstm_cell_23/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_23/lstm_cell_23/bias/v/Read/ReadVariableOpConst*I
TinB
@2>	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *(
f#R!
__inference__traced_save_686171
 
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_19/kerneldense_19/biasdense_20/kerneldense_20/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_21/lstm_cell_21/kernel%lstm_21/lstm_cell_21/recurrent_kernellstm_21/lstm_cell_21/biaslstm_22/lstm_cell_22/kernel%lstm_22/lstm_cell_22/recurrent_kernellstm_22/lstm_cell_22/biaslstm_23/lstm_cell_23/kernel%lstm_23/lstm_cell_23/recurrent_kernellstm_23/lstm_cell_23/biastotalcounttotal_1count_1Adam/dense_19/kernel/mAdam/dense_19/bias/mAdam/dense_20/kernel/mAdam/dense_20/bias/mAdam/dense_21/kernel/mAdam/dense_21/bias/mAdam/dense_22/kernel/mAdam/dense_22/bias/m"Adam/lstm_21/lstm_cell_21/kernel/m,Adam/lstm_21/lstm_cell_21/recurrent_kernel/m Adam/lstm_21/lstm_cell_21/bias/m"Adam/lstm_22/lstm_cell_22/kernel/m,Adam/lstm_22/lstm_cell_22/recurrent_kernel/m Adam/lstm_22/lstm_cell_22/bias/m"Adam/lstm_23/lstm_cell_23/kernel/m,Adam/lstm_23/lstm_cell_23/recurrent_kernel/m Adam/lstm_23/lstm_cell_23/bias/mAdam/dense_19/kernel/vAdam/dense_19/bias/vAdam/dense_20/kernel/vAdam/dense_20/bias/vAdam/dense_21/kernel/vAdam/dense_21/bias/vAdam/dense_22/kernel/vAdam/dense_22/bias/v"Adam/lstm_21/lstm_cell_21/kernel/v,Adam/lstm_21/lstm_cell_21/recurrent_kernel/v Adam/lstm_21/lstm_cell_21/bias/v"Adam/lstm_22/lstm_cell_22/kernel/v,Adam/lstm_22/lstm_cell_22/recurrent_kernel/v Adam/lstm_22/lstm_cell_22/bias/v"Adam/lstm_23/lstm_cell_23/kernel/v,Adam/lstm_23/lstm_cell_23/recurrent_kernel/v Adam/lstm_23/lstm_cell_23/bias/v*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *+
f&R$
"__inference__traced_restore_686361┼ѕ1
П
╗
-__inference_sequential_7_layer_call_fn_682596

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:
identityѕбStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *3
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_681530o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
 7
╩
while_body_684974
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_23_matmul_readvariableop_resource_0:@G
5while_lstm_cell_23_matmul_1_readvariableop_resource_0:@B
4while_lstm_cell_23_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_23_matmul_readvariableop_resource:@E
3while_lstm_cell_23_matmul_1_readvariableop_resource:@@
2while_lstm_cell_23_biasadd_readvariableop_resource:@ѕб)while/lstm_cell_23/BiasAdd/ReadVariableOpб(while/lstm_cell_23/MatMul/ReadVariableOpб*while/lstm_cell_23/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ю
(while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_23_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0╣
while/lstm_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @а
*while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_23_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0а
while/lstm_cell_23/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_23/addAddV2#while/lstm_cell_23/MatMul:product:0%while/lstm_cell_23/MatMul_1:product:0*
T0*'
_output_shapes
:         @џ
)while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0д
while/lstm_cell_23/BiasAddBiasAddwhile/lstm_cell_23/add:z:01while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
"while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_23/splitSplit+while/lstm_cell_23/split/split_dim:output:0#while/lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitz
while/lstm_cell_23/SigmoidSigmoid!while/lstm_cell_23/split:output:0*
T0*'
_output_shapes
:         |
while/lstm_cell_23/Sigmoid_1Sigmoid!while/lstm_cell_23/split:output:1*
T0*'
_output_shapes
:         є
while/lstm_cell_23/mulMul while/lstm_cell_23/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         t
while/lstm_cell_23/ReluRelu!while/lstm_cell_23/split:output:2*
T0*'
_output_shapes
:         ў
while/lstm_cell_23/mul_1Mulwhile/lstm_cell_23/Sigmoid:y:0%while/lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         Ї
while/lstm_cell_23/add_1AddV2while/lstm_cell_23/mul:z:0while/lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         |
while/lstm_cell_23/Sigmoid_2Sigmoid!while/lstm_cell_23/split:output:3*
T0*'
_output_shapes
:         q
while/lstm_cell_23/Relu_1Reluwhile/lstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         ю
while/lstm_cell_23/mul_2Mul while/lstm_cell_23/Sigmoid_2:y:0'while/lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_23/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_23/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         y
while/Identity_5Identitywhile/lstm_cell_23/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         л

while/NoOpNoOp*^while/lstm_cell_23/BiasAdd/ReadVariableOp)^while/lstm_cell_23/MatMul/ReadVariableOp+^while/lstm_cell_23/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_23_biasadd_readvariableop_resource4while_lstm_cell_23_biasadd_readvariableop_resource_0"l
3while_lstm_cell_23_matmul_1_readvariableop_resource5while_lstm_cell_23_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_23_matmul_readvariableop_resource3while_lstm_cell_23_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2V
)while/lstm_cell_23/BiasAdd/ReadVariableOp)while/lstm_cell_23/BiasAdd/ReadVariableOp2T
(while/lstm_cell_23/MatMul/ReadVariableOp(while/lstm_cell_23/MatMul/ReadVariableOp2X
*while/lstm_cell_23/MatMul_1/ReadVariableOp*while/lstm_cell_23/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
ћ

e
F__inference_dropout_33_layer_call_and_return_conditional_losses_684228

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:љ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=ф
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         ]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         :S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ЎJ
ў
C__inference_lstm_21_layer_call_and_return_conditional_losses_682231

inputs=
+lstm_cell_21_matmul_readvariableop_resource:@?
-lstm_cell_21_matmul_1_readvariableop_resource:@:
,lstm_cell_21_biasadd_readvariableop_resource:@
identityѕб#lstm_cell_21/BiasAdd/ReadVariableOpб"lstm_cell_21/MatMul/ReadVariableOpб$lstm_cell_21/MatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskј
"lstm_cell_21/MatMul/ReadVariableOpReadVariableOp+lstm_cell_21_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ћ
lstm_cell_21/MatMulMatMulstrided_slice_2:output:0*lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @њ
$lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_21_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell_21/MatMul_1MatMulzeros:output:0,lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @І
lstm_cell_21/addAddV2lstm_cell_21/MatMul:product:0lstm_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:         @ї
#lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ћ
lstm_cell_21/BiasAddBiasAddlstm_cell_21/add:z:0+lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @^
lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_21/splitSplit%lstm_cell_21/split/split_dim:output:0lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitn
lstm_cell_21/SigmoidSigmoidlstm_cell_21/split:output:0*
T0*'
_output_shapes
:         p
lstm_cell_21/Sigmoid_1Sigmoidlstm_cell_21/split:output:1*
T0*'
_output_shapes
:         w
lstm_cell_21/mulMullstm_cell_21/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         h
lstm_cell_21/ReluRelulstm_cell_21/split:output:2*
T0*'
_output_shapes
:         є
lstm_cell_21/mul_1Mullstm_cell_21/Sigmoid:y:0lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         {
lstm_cell_21/add_1AddV2lstm_cell_21/mul:z:0lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         p
lstm_cell_21/Sigmoid_2Sigmoidlstm_cell_21/split:output:3*
T0*'
_output_shapes
:         e
lstm_cell_21/Relu_1Relulstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         і
lstm_cell_21/mul_2Mullstm_cell_21/Sigmoid_2:y:0!lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѓ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_21_matmul_readvariableop_resource-lstm_cell_21_matmul_1_readvariableop_resource,lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_682147*
condR
while_cond_682146*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         └
NoOpNoOp$^lstm_cell_21/BiasAdd/ReadVariableOp#^lstm_cell_21/MatMul/ReadVariableOp%^lstm_cell_21/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2J
#lstm_cell_21/BiasAdd/ReadVariableOp#lstm_cell_21/BiasAdd/ReadVariableOp2H
"lstm_cell_21/MatMul/ReadVariableOp"lstm_cell_21/MatMul/ReadVariableOp2L
$lstm_cell_21/MatMul_1/ReadVariableOp$lstm_cell_21/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ъJ
ў
C__inference_lstm_23_layer_call_and_return_conditional_losses_681855

inputs=
+lstm_cell_23_matmul_readvariableop_resource:@?
-lstm_cell_23_matmul_1_readvariableop_resource:@:
,lstm_cell_23_biasadd_readvariableop_resource:@
identityѕб#lstm_cell_23/BiasAdd/ReadVariableOpб"lstm_cell_23/MatMul/ReadVariableOpб$lstm_cell_23/MatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskј
"lstm_cell_23/MatMul/ReadVariableOpReadVariableOp+lstm_cell_23_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ћ
lstm_cell_23/MatMulMatMulstrided_slice_2:output:0*lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @њ
$lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_23_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell_23/MatMul_1MatMulzeros:output:0,lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @І
lstm_cell_23/addAddV2lstm_cell_23/MatMul:product:0lstm_cell_23/MatMul_1:product:0*
T0*'
_output_shapes
:         @ї
#lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ћ
lstm_cell_23/BiasAddBiasAddlstm_cell_23/add:z:0+lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @^
lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_23/splitSplit%lstm_cell_23/split/split_dim:output:0lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitn
lstm_cell_23/SigmoidSigmoidlstm_cell_23/split:output:0*
T0*'
_output_shapes
:         p
lstm_cell_23/Sigmoid_1Sigmoidlstm_cell_23/split:output:1*
T0*'
_output_shapes
:         w
lstm_cell_23/mulMullstm_cell_23/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         h
lstm_cell_23/ReluRelulstm_cell_23/split:output:2*
T0*'
_output_shapes
:         є
lstm_cell_23/mul_1Mullstm_cell_23/Sigmoid:y:0lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         {
lstm_cell_23/add_1AddV2lstm_cell_23/mul:z:0lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         p
lstm_cell_23/Sigmoid_2Sigmoidlstm_cell_23/split:output:3*
T0*'
_output_shapes
:         e
lstm_cell_23/Relu_1Relulstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         і
lstm_cell_23/mul_2Mullstm_cell_23/Sigmoid_2:y:0!lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѓ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_23_matmul_readvariableop_resource-lstm_cell_23_matmul_1_readvariableop_resource,lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_681771*
condR
while_cond_681770*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         └
NoOpNoOp$^lstm_cell_23/BiasAdd/ReadVariableOp#^lstm_cell_23/MatMul/ReadVariableOp%^lstm_cell_23/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2J
#lstm_cell_23/BiasAdd/ReadVariableOp#lstm_cell_23/BiasAdd/ReadVariableOp2H
"lstm_cell_23/MatMul/ReadVariableOp"lstm_cell_23/MatMul/ReadVariableOp2L
$lstm_cell_23/MatMul_1/ReadVariableOp$lstm_cell_23/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ј8
ѓ
C__inference_lstm_23_layer_call_and_return_conditional_losses_680954

inputs%
lstm_cell_23_680872:@%
lstm_cell_23_680874:@!
lstm_cell_23_680876:@
identityѕб$lstm_cell_23/StatefulPartitionedCallбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЭ
$lstm_cell_23/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_23_680872lstm_cell_23_680874lstm_cell_23_680876*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_lstm_cell_23_layer_call_and_return_conditional_losses_680826n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : и
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_23_680872lstm_cell_23_680874lstm_cell_23_680876*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_680885*
condR
while_cond_680884*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         u
NoOpNoOp%^lstm_cell_23/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2L
$lstm_cell_23/StatefulPartitionedCall$lstm_cell_23/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
ћ

e
F__inference_dropout_34_layer_call_and_return_conditional_losses_684871

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:љ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=ф
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         ]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         :S O
+
_output_shapes
:         
 
_user_specified_nameinputs
 7
╩
while_body_684331
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_22_matmul_readvariableop_resource_0:@G
5while_lstm_cell_22_matmul_1_readvariableop_resource_0:@B
4while_lstm_cell_22_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_22_matmul_readvariableop_resource:@E
3while_lstm_cell_22_matmul_1_readvariableop_resource:@@
2while_lstm_cell_22_biasadd_readvariableop_resource:@ѕб)while/lstm_cell_22/BiasAdd/ReadVariableOpб(while/lstm_cell_22/MatMul/ReadVariableOpб*while/lstm_cell_22/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ю
(while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_22_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0╣
while/lstm_cell_22/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @а
*while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_22_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0а
while/lstm_cell_22/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_22/addAddV2#while/lstm_cell_22/MatMul:product:0%while/lstm_cell_22/MatMul_1:product:0*
T0*'
_output_shapes
:         @џ
)while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0д
while/lstm_cell_22/BiasAddBiasAddwhile/lstm_cell_22/add:z:01while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0#while/lstm_cell_22/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitz
while/lstm_cell_22/SigmoidSigmoid!while/lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         |
while/lstm_cell_22/Sigmoid_1Sigmoid!while/lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         є
while/lstm_cell_22/mulMul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         t
while/lstm_cell_22/ReluRelu!while/lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         ў
while/lstm_cell_22/mul_1Mulwhile/lstm_cell_22/Sigmoid:y:0%while/lstm_cell_22/Relu:activations:0*
T0*'
_output_shapes
:         Ї
while/lstm_cell_22/add_1AddV2while/lstm_cell_22/mul:z:0while/lstm_cell_22/mul_1:z:0*
T0*'
_output_shapes
:         |
while/lstm_cell_22/Sigmoid_2Sigmoid!while/lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         q
while/lstm_cell_22/Relu_1Reluwhile/lstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         ю
while/lstm_cell_22/mul_2Mul while/lstm_cell_22/Sigmoid_2:y:0'while/lstm_cell_22/Relu_1:activations:0*
T0*'
_output_shapes
:         ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_22/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_22/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         y
while/Identity_5Identitywhile/lstm_cell_22/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         л

while/NoOpNoOp*^while/lstm_cell_22/BiasAdd/ReadVariableOp)^while/lstm_cell_22/MatMul/ReadVariableOp+^while/lstm_cell_22/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_22_biasadd_readvariableop_resource4while_lstm_cell_22_biasadd_readvariableop_resource_0"l
3while_lstm_cell_22_matmul_1_readvariableop_resource5while_lstm_cell_22_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_22_matmul_readvariableop_resource3while_lstm_cell_22_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2V
)while/lstm_cell_22/BiasAdd/ReadVariableOp)while/lstm_cell_22/BiasAdd/ReadVariableOp2T
(while/lstm_cell_22/MatMul/ReadVariableOp(while/lstm_cell_22/MatMul/ReadVariableOp2X
*while/lstm_cell_22/MatMul_1/ReadVariableOp*while/lstm_cell_22/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
ъJ
ў
C__inference_lstm_23_layer_call_and_return_conditional_losses_685487

inputs=
+lstm_cell_23_matmul_readvariableop_resource:@?
-lstm_cell_23_matmul_1_readvariableop_resource:@:
,lstm_cell_23_biasadd_readvariableop_resource:@
identityѕб#lstm_cell_23/BiasAdd/ReadVariableOpб"lstm_cell_23/MatMul/ReadVariableOpб$lstm_cell_23/MatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskј
"lstm_cell_23/MatMul/ReadVariableOpReadVariableOp+lstm_cell_23_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ћ
lstm_cell_23/MatMulMatMulstrided_slice_2:output:0*lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @њ
$lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_23_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell_23/MatMul_1MatMulzeros:output:0,lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @І
lstm_cell_23/addAddV2lstm_cell_23/MatMul:product:0lstm_cell_23/MatMul_1:product:0*
T0*'
_output_shapes
:         @ї
#lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ћ
lstm_cell_23/BiasAddBiasAddlstm_cell_23/add:z:0+lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @^
lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_23/splitSplit%lstm_cell_23/split/split_dim:output:0lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitn
lstm_cell_23/SigmoidSigmoidlstm_cell_23/split:output:0*
T0*'
_output_shapes
:         p
lstm_cell_23/Sigmoid_1Sigmoidlstm_cell_23/split:output:1*
T0*'
_output_shapes
:         w
lstm_cell_23/mulMullstm_cell_23/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         h
lstm_cell_23/ReluRelulstm_cell_23/split:output:2*
T0*'
_output_shapes
:         є
lstm_cell_23/mul_1Mullstm_cell_23/Sigmoid:y:0lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         {
lstm_cell_23/add_1AddV2lstm_cell_23/mul:z:0lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         p
lstm_cell_23/Sigmoid_2Sigmoidlstm_cell_23/split:output:3*
T0*'
_output_shapes
:         e
lstm_cell_23/Relu_1Relulstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         і
lstm_cell_23/mul_2Mullstm_cell_23/Sigmoid_2:y:0!lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѓ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_23_matmul_readvariableop_resource-lstm_cell_23_matmul_1_readvariableop_resource,lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_685403*
condR
while_cond_685402*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         └
NoOpNoOp$^lstm_cell_23/BiasAdd/ReadVariableOp#^lstm_cell_23/MatMul/ReadVariableOp%^lstm_cell_23/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2J
#lstm_cell_23/BiasAdd/ReadVariableOp#lstm_cell_23/BiasAdd/ReadVariableOp2H
"lstm_cell_23/MatMul/ReadVariableOp"lstm_cell_23/MatMul/ReadVariableOp2L
$lstm_cell_23/MatMul_1/ReadVariableOp$lstm_cell_23/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ИA
╩

lstm_23_while_body_682974,
(lstm_23_while_lstm_23_while_loop_counter2
.lstm_23_while_lstm_23_while_maximum_iterations
lstm_23_while_placeholder
lstm_23_while_placeholder_1
lstm_23_while_placeholder_2
lstm_23_while_placeholder_3+
'lstm_23_while_lstm_23_strided_slice_1_0g
clstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensor_0M
;lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0:@O
=lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0:@J
<lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0:@
lstm_23_while_identity
lstm_23_while_identity_1
lstm_23_while_identity_2
lstm_23_while_identity_3
lstm_23_while_identity_4
lstm_23_while_identity_5)
%lstm_23_while_lstm_23_strided_slice_1e
alstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensorK
9lstm_23_while_lstm_cell_23_matmul_readvariableop_resource:@M
;lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource:@H
:lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource:@ѕб1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOpб0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOpб2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOpљ
?lstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╬
1lstm_23/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensor_0lstm_23_while_placeholderHlstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0г
0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp;lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0Л
!lstm_23/while/lstm_cell_23/MatMulMatMul8lstm_23/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @░
2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp=lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0И
#lstm_23/while/lstm_cell_23/MatMul_1MatMullstm_23_while_placeholder_2:lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @х
lstm_23/while/lstm_cell_23/addAddV2+lstm_23/while/lstm_cell_23/MatMul:product:0-lstm_23/while/lstm_cell_23/MatMul_1:product:0*
T0*'
_output_shapes
:         @ф
1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp<lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Й
"lstm_23/while/lstm_cell_23/BiasAddBiasAdd"lstm_23/while/lstm_cell_23/add:z:09lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @l
*lstm_23/while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Є
 lstm_23/while/lstm_cell_23/splitSplit3lstm_23/while/lstm_cell_23/split/split_dim:output:0+lstm_23/while/lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitі
"lstm_23/while/lstm_cell_23/SigmoidSigmoid)lstm_23/while/lstm_cell_23/split:output:0*
T0*'
_output_shapes
:         ї
$lstm_23/while/lstm_cell_23/Sigmoid_1Sigmoid)lstm_23/while/lstm_cell_23/split:output:1*
T0*'
_output_shapes
:         ъ
lstm_23/while/lstm_cell_23/mulMul(lstm_23/while/lstm_cell_23/Sigmoid_1:y:0lstm_23_while_placeholder_3*
T0*'
_output_shapes
:         ё
lstm_23/while/lstm_cell_23/ReluRelu)lstm_23/while/lstm_cell_23/split:output:2*
T0*'
_output_shapes
:         ░
 lstm_23/while/lstm_cell_23/mul_1Mul&lstm_23/while/lstm_cell_23/Sigmoid:y:0-lstm_23/while/lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         Ц
 lstm_23/while/lstm_cell_23/add_1AddV2"lstm_23/while/lstm_cell_23/mul:z:0$lstm_23/while/lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         ї
$lstm_23/while/lstm_cell_23/Sigmoid_2Sigmoid)lstm_23/while/lstm_cell_23/split:output:3*
T0*'
_output_shapes
:         Ђ
!lstm_23/while/lstm_cell_23/Relu_1Relu$lstm_23/while/lstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         ┤
 lstm_23/while/lstm_cell_23/mul_2Mul(lstm_23/while/lstm_cell_23/Sigmoid_2:y:0/lstm_23/while/lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         т
2lstm_23/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_23_while_placeholder_1lstm_23_while_placeholder$lstm_23/while/lstm_cell_23/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмU
lstm_23/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_23/while/addAddV2lstm_23_while_placeholderlstm_23/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_23/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Є
lstm_23/while/add_1AddV2(lstm_23_while_lstm_23_while_loop_counterlstm_23/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_23/while/IdentityIdentitylstm_23/while/add_1:z:0^lstm_23/while/NoOp*
T0*
_output_shapes
: і
lstm_23/while/Identity_1Identity.lstm_23_while_lstm_23_while_maximum_iterations^lstm_23/while/NoOp*
T0*
_output_shapes
: q
lstm_23/while/Identity_2Identitylstm_23/while/add:z:0^lstm_23/while/NoOp*
T0*
_output_shapes
: ъ
lstm_23/while/Identity_3IdentityBlstm_23/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_23/while/NoOp*
T0*
_output_shapes
: Љ
lstm_23/while/Identity_4Identity$lstm_23/while/lstm_cell_23/mul_2:z:0^lstm_23/while/NoOp*
T0*'
_output_shapes
:         Љ
lstm_23/while/Identity_5Identity$lstm_23/while/lstm_cell_23/add_1:z:0^lstm_23/while/NoOp*
T0*'
_output_shapes
:         ­
lstm_23/while/NoOpNoOp2^lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp1^lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp3^lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_23_while_identitylstm_23/while/Identity:output:0"=
lstm_23_while_identity_1!lstm_23/while/Identity_1:output:0"=
lstm_23_while_identity_2!lstm_23/while/Identity_2:output:0"=
lstm_23_while_identity_3!lstm_23/while/Identity_3:output:0"=
lstm_23_while_identity_4!lstm_23/while/Identity_4:output:0"=
lstm_23_while_identity_5!lstm_23/while/Identity_5:output:0"P
%lstm_23_while_lstm_23_strided_slice_1'lstm_23_while_lstm_23_strided_slice_1_0"z
:lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource<lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0"|
;lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource=lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0"x
9lstm_23_while_lstm_cell_23_matmul_readvariableop_resource;lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0"╚
alstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensorclstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2f
1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp2d
0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp2h
2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
┘
d
F__inference_dropout_38_layer_call_and_return_conditional_losses_685643

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
 7
╩
while_body_682147
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_21_matmul_readvariableop_resource_0:@G
5while_lstm_cell_21_matmul_1_readvariableop_resource_0:@B
4while_lstm_cell_21_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_21_matmul_readvariableop_resource:@E
3while_lstm_cell_21_matmul_1_readvariableop_resource:@@
2while_lstm_cell_21_biasadd_readvariableop_resource:@ѕб)while/lstm_cell_21/BiasAdd/ReadVariableOpб(while/lstm_cell_21/MatMul/ReadVariableOpб*while/lstm_cell_21/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ю
(while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_21_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0╣
while/lstm_cell_21/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @а
*while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_21_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0а
while/lstm_cell_21/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_21/addAddV2#while/lstm_cell_21/MatMul:product:0%while/lstm_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:         @џ
)while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0д
while/lstm_cell_21/BiasAddBiasAddwhile/lstm_cell_21/add:z:01while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
"while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_21/splitSplit+while/lstm_cell_21/split/split_dim:output:0#while/lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitz
while/lstm_cell_21/SigmoidSigmoid!while/lstm_cell_21/split:output:0*
T0*'
_output_shapes
:         |
while/lstm_cell_21/Sigmoid_1Sigmoid!while/lstm_cell_21/split:output:1*
T0*'
_output_shapes
:         є
while/lstm_cell_21/mulMul while/lstm_cell_21/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         t
while/lstm_cell_21/ReluRelu!while/lstm_cell_21/split:output:2*
T0*'
_output_shapes
:         ў
while/lstm_cell_21/mul_1Mulwhile/lstm_cell_21/Sigmoid:y:0%while/lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         Ї
while/lstm_cell_21/add_1AddV2while/lstm_cell_21/mul:z:0while/lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         |
while/lstm_cell_21/Sigmoid_2Sigmoid!while/lstm_cell_21/split:output:3*
T0*'
_output_shapes
:         q
while/lstm_cell_21/Relu_1Reluwhile/lstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         ю
while/lstm_cell_21/mul_2Mul while/lstm_cell_21/Sigmoid_2:y:0'while/lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_21/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_21/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         y
while/Identity_5Identitywhile/lstm_cell_21/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         л

while/NoOpNoOp*^while/lstm_cell_21/BiasAdd/ReadVariableOp)^while/lstm_cell_21/MatMul/ReadVariableOp+^while/lstm_cell_21/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_21_biasadd_readvariableop_resource4while_lstm_cell_21_biasadd_readvariableop_resource_0"l
3while_lstm_cell_21_matmul_1_readvariableop_resource5while_lstm_cell_21_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_21_matmul_readvariableop_resource3while_lstm_cell_21_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2V
)while/lstm_cell_21/BiasAdd/ReadVariableOp)while/lstm_cell_21/BiasAdd/ReadVariableOp2T
(while/lstm_cell_21/MatMul/ReadVariableOp(while/lstm_cell_21/MatMul/ReadVariableOp2X
*while/lstm_cell_21/MatMul_1/ReadVariableOp*while/lstm_cell_21/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
 7
╩
while_body_684760
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_22_matmul_readvariableop_resource_0:@G
5while_lstm_cell_22_matmul_1_readvariableop_resource_0:@B
4while_lstm_cell_22_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_22_matmul_readvariableop_resource:@E
3while_lstm_cell_22_matmul_1_readvariableop_resource:@@
2while_lstm_cell_22_biasadd_readvariableop_resource:@ѕб)while/lstm_cell_22/BiasAdd/ReadVariableOpб(while/lstm_cell_22/MatMul/ReadVariableOpб*while/lstm_cell_22/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ю
(while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_22_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0╣
while/lstm_cell_22/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @а
*while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_22_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0а
while/lstm_cell_22/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_22/addAddV2#while/lstm_cell_22/MatMul:product:0%while/lstm_cell_22/MatMul_1:product:0*
T0*'
_output_shapes
:         @џ
)while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0д
while/lstm_cell_22/BiasAddBiasAddwhile/lstm_cell_22/add:z:01while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0#while/lstm_cell_22/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitz
while/lstm_cell_22/SigmoidSigmoid!while/lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         |
while/lstm_cell_22/Sigmoid_1Sigmoid!while/lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         є
while/lstm_cell_22/mulMul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         t
while/lstm_cell_22/ReluRelu!while/lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         ў
while/lstm_cell_22/mul_1Mulwhile/lstm_cell_22/Sigmoid:y:0%while/lstm_cell_22/Relu:activations:0*
T0*'
_output_shapes
:         Ї
while/lstm_cell_22/add_1AddV2while/lstm_cell_22/mul:z:0while/lstm_cell_22/mul_1:z:0*
T0*'
_output_shapes
:         |
while/lstm_cell_22/Sigmoid_2Sigmoid!while/lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         q
while/lstm_cell_22/Relu_1Reluwhile/lstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         ю
while/lstm_cell_22/mul_2Mul while/lstm_cell_22/Sigmoid_2:y:0'while/lstm_cell_22/Relu_1:activations:0*
T0*'
_output_shapes
:         ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_22/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_22/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         y
while/Identity_5Identitywhile/lstm_cell_22/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         л

while/NoOpNoOp*^while/lstm_cell_22/BiasAdd/ReadVariableOp)^while/lstm_cell_22/MatMul/ReadVariableOp+^while/lstm_cell_22/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_22_biasadd_readvariableop_resource4while_lstm_cell_22_biasadd_readvariableop_resource_0"l
3while_lstm_cell_22_matmul_1_readvariableop_resource5while_lstm_cell_22_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_22_matmul_readvariableop_resource3while_lstm_cell_22_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2V
)while/lstm_cell_22/BiasAdd/ReadVariableOp)while/lstm_cell_22/BiasAdd/ReadVariableOp2T
(while/lstm_cell_22/MatMul/ReadVariableOp(while/lstm_cell_22/MatMul/ReadVariableOp2X
*while/lstm_cell_22/MatMul_1/ReadVariableOp*while/lstm_cell_22/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
┤
G
+__inference_dropout_34_layer_call_fn_684849

inputs
identityИ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_34_layer_call_and_return_conditional_losses_681282d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         :S O
+
_output_shapes
:         
 
_user_specified_nameinputs
┘
d
F__inference_dropout_36_layer_call_and_return_conditional_losses_685549

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ш
d
+__inference_dropout_35_layer_call_fn_685497

inputs
identityѕбStatefulPartitionedCall─
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_35_layer_call_and_return_conditional_losses_681696o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┘
d
F__inference_dropout_38_layer_call_and_return_conditional_losses_681511

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ОJ
џ
C__inference_lstm_21_layer_call_and_return_conditional_losses_683915
inputs_0=
+lstm_cell_21_matmul_readvariableop_resource:@?
-lstm_cell_21_matmul_1_readvariableop_resource:@:
,lstm_cell_21_biasadd_readvariableop_resource:@
identityѕб#lstm_cell_21/BiasAdd/ReadVariableOpб"lstm_cell_21/MatMul/ReadVariableOpб$lstm_cell_21/MatMul_1/ReadVariableOpбwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskј
"lstm_cell_21/MatMul/ReadVariableOpReadVariableOp+lstm_cell_21_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ћ
lstm_cell_21/MatMulMatMulstrided_slice_2:output:0*lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @њ
$lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_21_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell_21/MatMul_1MatMulzeros:output:0,lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @І
lstm_cell_21/addAddV2lstm_cell_21/MatMul:product:0lstm_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:         @ї
#lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ћ
lstm_cell_21/BiasAddBiasAddlstm_cell_21/add:z:0+lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @^
lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_21/splitSplit%lstm_cell_21/split/split_dim:output:0lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitn
lstm_cell_21/SigmoidSigmoidlstm_cell_21/split:output:0*
T0*'
_output_shapes
:         p
lstm_cell_21/Sigmoid_1Sigmoidlstm_cell_21/split:output:1*
T0*'
_output_shapes
:         w
lstm_cell_21/mulMullstm_cell_21/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         h
lstm_cell_21/ReluRelulstm_cell_21/split:output:2*
T0*'
_output_shapes
:         є
lstm_cell_21/mul_1Mullstm_cell_21/Sigmoid:y:0lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         {
lstm_cell_21/add_1AddV2lstm_cell_21/mul:z:0lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         p
lstm_cell_21/Sigmoid_2Sigmoidlstm_cell_21/split:output:3*
T0*'
_output_shapes
:         e
lstm_cell_21/Relu_1Relulstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         і
lstm_cell_21/mul_2Mullstm_cell_21/Sigmoid_2:y:0!lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѓ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_21_matmul_readvariableop_resource-lstm_cell_21_matmul_1_readvariableop_resource,lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_683831*
condR
while_cond_683830*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  └
NoOpNoOp$^lstm_cell_21/BiasAdd/ReadVariableOp#^lstm_cell_21/MatMul/ReadVariableOp%^lstm_cell_21/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2J
#lstm_cell_21/BiasAdd/ReadVariableOp#lstm_cell_21/BiasAdd/ReadVariableOp2H
"lstm_cell_21/MatMul/ReadVariableOp"lstm_cell_21/MatMul/ReadVariableOp2L
$lstm_cell_21/MatMul_1/ReadVariableOp$lstm_cell_21/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
┘
d
F__inference_dropout_37_layer_call_and_return_conditional_losses_681487

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ќ

с
lstm_22_while_cond_682833,
(lstm_22_while_lstm_22_while_loop_counter2
.lstm_22_while_lstm_22_while_maximum_iterations
lstm_22_while_placeholder
lstm_22_while_placeholder_1
lstm_22_while_placeholder_2
lstm_22_while_placeholder_3.
*lstm_22_while_less_lstm_22_strided_slice_1D
@lstm_22_while_lstm_22_while_cond_682833___redundant_placeholder0D
@lstm_22_while_lstm_22_while_cond_682833___redundant_placeholder1D
@lstm_22_while_lstm_22_while_cond_682833___redundant_placeholder2D
@lstm_22_while_lstm_22_while_cond_682833___redundant_placeholder3
lstm_22_while_identity
ѓ
lstm_22/while/LessLesslstm_22_while_placeholder*lstm_22_while_less_lstm_22_strided_slice_1*
T0*
_output_shapes
: [
lstm_22/while/IdentityIdentitylstm_22/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_22_while_identitylstm_22/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
┼
ќ
)__inference_dense_22_layer_call_fn_685664

inputs
unknown:
	unknown_0:
identityѕбStatefulPartitionedCall▄
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_681523o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
х
├
while_cond_679993
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_679993___redundant_placeholder04
0while_while_cond_679993___redundant_placeholder14
0while_while_cond_679993___redundant_placeholder24
0while_while_cond_679993___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
ѓ
у
&sequential_7_lstm_23_while_cond_679797F
Bsequential_7_lstm_23_while_sequential_7_lstm_23_while_loop_counterL
Hsequential_7_lstm_23_while_sequential_7_lstm_23_while_maximum_iterations*
&sequential_7_lstm_23_while_placeholder,
(sequential_7_lstm_23_while_placeholder_1,
(sequential_7_lstm_23_while_placeholder_2,
(sequential_7_lstm_23_while_placeholder_3H
Dsequential_7_lstm_23_while_less_sequential_7_lstm_23_strided_slice_1^
Zsequential_7_lstm_23_while_sequential_7_lstm_23_while_cond_679797___redundant_placeholder0^
Zsequential_7_lstm_23_while_sequential_7_lstm_23_while_cond_679797___redundant_placeholder1^
Zsequential_7_lstm_23_while_sequential_7_lstm_23_while_cond_679797___redundant_placeholder2^
Zsequential_7_lstm_23_while_sequential_7_lstm_23_while_cond_679797___redundant_placeholder3'
#sequential_7_lstm_23_while_identity
Х
sequential_7/lstm_23/while/LessLess&sequential_7_lstm_23_while_placeholderDsequential_7_lstm_23_while_less_sequential_7_lstm_23_strided_slice_1*
T0*
_output_shapes
: u
#sequential_7/lstm_23/while/IdentityIdentity#sequential_7/lstm_23/while/Less:z:0*
T0
*
_output_shapes
: "S
#sequential_7_lstm_23_while_identity,sequential_7/lstm_23/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
х
├
while_cond_680693
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_680693___redundant_placeholder04
0while_while_cond_680693___redundant_placeholder14
0while_while_cond_680693___redundant_placeholder24
0while_while_cond_680693___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
Џ

ш
D__inference_dense_21_layer_call_and_return_conditional_losses_685628

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
 7
╩
while_body_681771
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_23_matmul_readvariableop_resource_0:@G
5while_lstm_cell_23_matmul_1_readvariableop_resource_0:@B
4while_lstm_cell_23_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_23_matmul_readvariableop_resource:@E
3while_lstm_cell_23_matmul_1_readvariableop_resource:@@
2while_lstm_cell_23_biasadd_readvariableop_resource:@ѕб)while/lstm_cell_23/BiasAdd/ReadVariableOpб(while/lstm_cell_23/MatMul/ReadVariableOpб*while/lstm_cell_23/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ю
(while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_23_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0╣
while/lstm_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @а
*while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_23_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0а
while/lstm_cell_23/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_23/addAddV2#while/lstm_cell_23/MatMul:product:0%while/lstm_cell_23/MatMul_1:product:0*
T0*'
_output_shapes
:         @џ
)while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0д
while/lstm_cell_23/BiasAddBiasAddwhile/lstm_cell_23/add:z:01while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
"while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_23/splitSplit+while/lstm_cell_23/split/split_dim:output:0#while/lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitz
while/lstm_cell_23/SigmoidSigmoid!while/lstm_cell_23/split:output:0*
T0*'
_output_shapes
:         |
while/lstm_cell_23/Sigmoid_1Sigmoid!while/lstm_cell_23/split:output:1*
T0*'
_output_shapes
:         є
while/lstm_cell_23/mulMul while/lstm_cell_23/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         t
while/lstm_cell_23/ReluRelu!while/lstm_cell_23/split:output:2*
T0*'
_output_shapes
:         ў
while/lstm_cell_23/mul_1Mulwhile/lstm_cell_23/Sigmoid:y:0%while/lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         Ї
while/lstm_cell_23/add_1AddV2while/lstm_cell_23/mul:z:0while/lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         |
while/lstm_cell_23/Sigmoid_2Sigmoid!while/lstm_cell_23/split:output:3*
T0*'
_output_shapes
:         q
while/lstm_cell_23/Relu_1Reluwhile/lstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         ю
while/lstm_cell_23/mul_2Mul while/lstm_cell_23/Sigmoid_2:y:0'while/lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_23/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_23/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         y
while/Identity_5Identitywhile/lstm_cell_23/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         л

while/NoOpNoOp*^while/lstm_cell_23/BiasAdd/ReadVariableOp)^while/lstm_cell_23/MatMul/ReadVariableOp+^while/lstm_cell_23/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_23_biasadd_readvariableop_resource4while_lstm_cell_23_biasadd_readvariableop_resource_0"l
3while_lstm_cell_23_matmul_1_readvariableop_resource5while_lstm_cell_23_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_23_matmul_readvariableop_resource3while_lstm_cell_23_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2V
)while/lstm_cell_23/BiasAdd/ReadVariableOp)while/lstm_cell_23/BiasAdd/ReadVariableOp2T
(while/lstm_cell_23/MatMul/ReadVariableOp(while/lstm_cell_23/MatMul/ReadVariableOp2X
*while/lstm_cell_23/MatMul_1/ReadVariableOp*while/lstm_cell_23/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
Ш
d
+__inference_dropout_36_layer_call_fn_685544

inputs
identityѕбStatefulPartitionedCall─
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_681663o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ЎJ
ў
C__inference_lstm_21_layer_call_and_return_conditional_losses_684058

inputs=
+lstm_cell_21_matmul_readvariableop_resource:@?
-lstm_cell_21_matmul_1_readvariableop_resource:@:
,lstm_cell_21_biasadd_readvariableop_resource:@
identityѕб#lstm_cell_21/BiasAdd/ReadVariableOpб"lstm_cell_21/MatMul/ReadVariableOpб$lstm_cell_21/MatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskј
"lstm_cell_21/MatMul/ReadVariableOpReadVariableOp+lstm_cell_21_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ћ
lstm_cell_21/MatMulMatMulstrided_slice_2:output:0*lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @њ
$lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_21_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell_21/MatMul_1MatMulzeros:output:0,lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @І
lstm_cell_21/addAddV2lstm_cell_21/MatMul:product:0lstm_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:         @ї
#lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ћ
lstm_cell_21/BiasAddBiasAddlstm_cell_21/add:z:0+lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @^
lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_21/splitSplit%lstm_cell_21/split/split_dim:output:0lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitn
lstm_cell_21/SigmoidSigmoidlstm_cell_21/split:output:0*
T0*'
_output_shapes
:         p
lstm_cell_21/Sigmoid_1Sigmoidlstm_cell_21/split:output:1*
T0*'
_output_shapes
:         w
lstm_cell_21/mulMullstm_cell_21/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         h
lstm_cell_21/ReluRelulstm_cell_21/split:output:2*
T0*'
_output_shapes
:         є
lstm_cell_21/mul_1Mullstm_cell_21/Sigmoid:y:0lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         {
lstm_cell_21/add_1AddV2lstm_cell_21/mul:z:0lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         p
lstm_cell_21/Sigmoid_2Sigmoidlstm_cell_21/split:output:3*
T0*'
_output_shapes
:         e
lstm_cell_21/Relu_1Relulstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         і
lstm_cell_21/mul_2Mullstm_cell_21/Sigmoid_2:y:0!lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѓ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_21_matmul_readvariableop_resource-lstm_cell_21_matmul_1_readvariableop_resource,lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_683974*
condR
while_cond_683973*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         └
NoOpNoOp$^lstm_cell_21/BiasAdd/ReadVariableOp#^lstm_cell_21/MatMul/ReadVariableOp%^lstm_cell_21/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2J
#lstm_cell_21/BiasAdd/ReadVariableOp#lstm_cell_21/BiasAdd/ReadVariableOp2H
"lstm_cell_21/MatMul/ReadVariableOp"lstm_cell_21/MatMul/ReadVariableOp2L
$lstm_cell_21/MatMul_1/ReadVariableOp$lstm_cell_21/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
 7
╩
while_body_684474
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_22_matmul_readvariableop_resource_0:@G
5while_lstm_cell_22_matmul_1_readvariableop_resource_0:@B
4while_lstm_cell_22_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_22_matmul_readvariableop_resource:@E
3while_lstm_cell_22_matmul_1_readvariableop_resource:@@
2while_lstm_cell_22_biasadd_readvariableop_resource:@ѕб)while/lstm_cell_22/BiasAdd/ReadVariableOpб(while/lstm_cell_22/MatMul/ReadVariableOpб*while/lstm_cell_22/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ю
(while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_22_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0╣
while/lstm_cell_22/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @а
*while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_22_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0а
while/lstm_cell_22/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_22/addAddV2#while/lstm_cell_22/MatMul:product:0%while/lstm_cell_22/MatMul_1:product:0*
T0*'
_output_shapes
:         @џ
)while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0д
while/lstm_cell_22/BiasAddBiasAddwhile/lstm_cell_22/add:z:01while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0#while/lstm_cell_22/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitz
while/lstm_cell_22/SigmoidSigmoid!while/lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         |
while/lstm_cell_22/Sigmoid_1Sigmoid!while/lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         є
while/lstm_cell_22/mulMul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         t
while/lstm_cell_22/ReluRelu!while/lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         ў
while/lstm_cell_22/mul_1Mulwhile/lstm_cell_22/Sigmoid:y:0%while/lstm_cell_22/Relu:activations:0*
T0*'
_output_shapes
:         Ї
while/lstm_cell_22/add_1AddV2while/lstm_cell_22/mul:z:0while/lstm_cell_22/mul_1:z:0*
T0*'
_output_shapes
:         |
while/lstm_cell_22/Sigmoid_2Sigmoid!while/lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         q
while/lstm_cell_22/Relu_1Reluwhile/lstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         ю
while/lstm_cell_22/mul_2Mul while/lstm_cell_22/Sigmoid_2:y:0'while/lstm_cell_22/Relu_1:activations:0*
T0*'
_output_shapes
:         ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_22/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_22/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         y
while/Identity_5Identitywhile/lstm_cell_22/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         л

while/NoOpNoOp*^while/lstm_cell_22/BiasAdd/ReadVariableOp)^while/lstm_cell_22/MatMul/ReadVariableOp+^while/lstm_cell_22/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_22_biasadd_readvariableop_resource4while_lstm_cell_22_biasadd_readvariableop_resource_0"l
3while_lstm_cell_22_matmul_1_readvariableop_resource5while_lstm_cell_22_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_22_matmul_readvariableop_resource3while_lstm_cell_22_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2V
)while/lstm_cell_22/BiasAdd/ReadVariableOp)while/lstm_cell_22/BiasAdd/ReadVariableOp2T
(while/lstm_cell_22/MatMul/ReadVariableOp(while/lstm_cell_22/MatMul/ReadVariableOp2X
*while/lstm_cell_22/MatMul_1/ReadVariableOp*while/lstm_cell_22/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
Р@
В
H__inference_sequential_7_layer_call_and_return_conditional_losses_682510
lstm_21_input 
lstm_21_682462:@ 
lstm_21_682464:@
lstm_21_682466:@ 
lstm_22_682470:@ 
lstm_22_682472:@
lstm_22_682474:@ 
lstm_23_682478:@ 
lstm_23_682480:@
lstm_23_682482:@!
dense_19_682486:
dense_19_682488:!
dense_20_682492:
dense_20_682494:!
dense_21_682498:
dense_21_682500:!
dense_22_682504:
dense_22_682506:
identityѕб dense_19/StatefulPartitionedCallб dense_20/StatefulPartitionedCallб dense_21/StatefulPartitionedCallб dense_22/StatefulPartitionedCallб"dropout_33/StatefulPartitionedCallб"dropout_34/StatefulPartitionedCallб"dropout_35/StatefulPartitionedCallб"dropout_36/StatefulPartitionedCallб"dropout_37/StatefulPartitionedCallб"dropout_38/StatefulPartitionedCallбlstm_21/StatefulPartitionedCallбlstm_22/StatefulPartitionedCallбlstm_23/StatefulPartitionedCallї
lstm_21/StatefulPartitionedCallStatefulPartitionedCalllstm_21_inputlstm_21_682462lstm_21_682464lstm_21_682466*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_lstm_21_layer_call_and_return_conditional_losses_682231ш
"dropout_33/StatefulPartitionedCallStatefulPartitionedCall(lstm_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_33_layer_call_and_return_conditional_losses_682072ф
lstm_22/StatefulPartitionedCallStatefulPartitionedCall+dropout_33/StatefulPartitionedCall:output:0lstm_22_682470lstm_22_682472lstm_22_682474*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_682043џ
"dropout_34/StatefulPartitionedCallStatefulPartitionedCall(lstm_22/StatefulPartitionedCall:output:0#^dropout_33/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_34_layer_call_and_return_conditional_losses_681884д
lstm_23/StatefulPartitionedCallStatefulPartitionedCall+dropout_34/StatefulPartitionedCall:output:0lstm_23_682478lstm_23_682480lstm_23_682482*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_lstm_23_layer_call_and_return_conditional_losses_681855ќ
"dropout_35/StatefulPartitionedCallStatefulPartitionedCall(lstm_23/StatefulPartitionedCall:output:0#^dropout_34/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_35_layer_call_and_return_conditional_losses_681696ў
 dense_19/StatefulPartitionedCallStatefulPartitionedCall+dropout_35/StatefulPartitionedCall:output:0dense_19_682486dense_19_682488*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_681452Ќ
"dropout_36/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0#^dropout_35/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_681663ў
 dense_20/StatefulPartitionedCallStatefulPartitionedCall+dropout_36/StatefulPartitionedCall:output:0dense_20_682492dense_20_682494*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_681476Ќ
"dropout_37/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0#^dropout_36/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_681630ў
 dense_21/StatefulPartitionedCallStatefulPartitionedCall+dropout_37/StatefulPartitionedCall:output:0dense_21_682498dense_21_682500*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_681500Ќ
"dropout_38/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0#^dropout_37/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_681597ў
 dense_22/StatefulPartitionedCallStatefulPartitionedCall+dropout_38/StatefulPartitionedCall:output:0dense_22_682504dense_22_682506*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_681523x
IdentityIdentity)dense_22/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ќ
NoOpNoOp!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall#^dropout_33/StatefulPartitionedCall#^dropout_34/StatefulPartitionedCall#^dropout_35/StatefulPartitionedCall#^dropout_36/StatefulPartitionedCall#^dropout_37/StatefulPartitionedCall#^dropout_38/StatefulPartitionedCall ^lstm_21/StatefulPartitionedCall ^lstm_22/StatefulPartitionedCall ^lstm_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         : : : : : : : : : : : : : : : : : 2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2H
"dropout_33/StatefulPartitionedCall"dropout_33/StatefulPartitionedCall2H
"dropout_34/StatefulPartitionedCall"dropout_34/StatefulPartitionedCall2H
"dropout_35/StatefulPartitionedCall"dropout_35/StatefulPartitionedCall2H
"dropout_36/StatefulPartitionedCall"dropout_36/StatefulPartitionedCall2H
"dropout_37/StatefulPartitionedCall"dropout_37/StatefulPartitionedCall2H
"dropout_38/StatefulPartitionedCall"dropout_38/StatefulPartitionedCall2B
lstm_21/StatefulPartitionedCalllstm_21/StatefulPartitionedCall2B
lstm_22/StatefulPartitionedCalllstm_22/StatefulPartitionedCall2B
lstm_23/StatefulPartitionedCalllstm_23/StatefulPartitionedCall:Z V
+
_output_shapes
:         
'
_user_specified_namelstm_21_input
Џ

ш
D__inference_dense_20_layer_call_and_return_conditional_losses_681476

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
═@
т
H__inference_sequential_7_layer_call_and_return_conditional_losses_682332

inputs 
lstm_21_682284:@ 
lstm_21_682286:@
lstm_21_682288:@ 
lstm_22_682292:@ 
lstm_22_682294:@
lstm_22_682296:@ 
lstm_23_682300:@ 
lstm_23_682302:@
lstm_23_682304:@!
dense_19_682308:
dense_19_682310:!
dense_20_682314:
dense_20_682316:!
dense_21_682320:
dense_21_682322:!
dense_22_682326:
dense_22_682328:
identityѕб dense_19/StatefulPartitionedCallб dense_20/StatefulPartitionedCallб dense_21/StatefulPartitionedCallб dense_22/StatefulPartitionedCallб"dropout_33/StatefulPartitionedCallб"dropout_34/StatefulPartitionedCallб"dropout_35/StatefulPartitionedCallб"dropout_36/StatefulPartitionedCallб"dropout_37/StatefulPartitionedCallб"dropout_38/StatefulPartitionedCallбlstm_21/StatefulPartitionedCallбlstm_22/StatefulPartitionedCallбlstm_23/StatefulPartitionedCallЁ
lstm_21/StatefulPartitionedCallStatefulPartitionedCallinputslstm_21_682284lstm_21_682286lstm_21_682288*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_lstm_21_layer_call_and_return_conditional_losses_682231ш
"dropout_33/StatefulPartitionedCallStatefulPartitionedCall(lstm_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_33_layer_call_and_return_conditional_losses_682072ф
lstm_22/StatefulPartitionedCallStatefulPartitionedCall+dropout_33/StatefulPartitionedCall:output:0lstm_22_682292lstm_22_682294lstm_22_682296*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_682043џ
"dropout_34/StatefulPartitionedCallStatefulPartitionedCall(lstm_22/StatefulPartitionedCall:output:0#^dropout_33/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_34_layer_call_and_return_conditional_losses_681884д
lstm_23/StatefulPartitionedCallStatefulPartitionedCall+dropout_34/StatefulPartitionedCall:output:0lstm_23_682300lstm_23_682302lstm_23_682304*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_lstm_23_layer_call_and_return_conditional_losses_681855ќ
"dropout_35/StatefulPartitionedCallStatefulPartitionedCall(lstm_23/StatefulPartitionedCall:output:0#^dropout_34/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_35_layer_call_and_return_conditional_losses_681696ў
 dense_19/StatefulPartitionedCallStatefulPartitionedCall+dropout_35/StatefulPartitionedCall:output:0dense_19_682308dense_19_682310*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_681452Ќ
"dropout_36/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0#^dropout_35/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_681663ў
 dense_20/StatefulPartitionedCallStatefulPartitionedCall+dropout_36/StatefulPartitionedCall:output:0dense_20_682314dense_20_682316*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_681476Ќ
"dropout_37/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0#^dropout_36/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_681630ў
 dense_21/StatefulPartitionedCallStatefulPartitionedCall+dropout_37/StatefulPartitionedCall:output:0dense_21_682320dense_21_682322*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_681500Ќ
"dropout_38/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0#^dropout_37/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_681597ў
 dense_22/StatefulPartitionedCallStatefulPartitionedCall+dropout_38/StatefulPartitionedCall:output:0dense_22_682326dense_22_682328*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_681523x
IdentityIdentity)dense_22/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ќ
NoOpNoOp!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall#^dropout_33/StatefulPartitionedCall#^dropout_34/StatefulPartitionedCall#^dropout_35/StatefulPartitionedCall#^dropout_36/StatefulPartitionedCall#^dropout_37/StatefulPartitionedCall#^dropout_38/StatefulPartitionedCall ^lstm_21/StatefulPartitionedCall ^lstm_22/StatefulPartitionedCall ^lstm_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         : : : : : : : : : : : : : : : : : 2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2H
"dropout_33/StatefulPartitionedCall"dropout_33/StatefulPartitionedCall2H
"dropout_34/StatefulPartitionedCall"dropout_34/StatefulPartitionedCall2H
"dropout_35/StatefulPartitionedCall"dropout_35/StatefulPartitionedCall2H
"dropout_36/StatefulPartitionedCall"dropout_36/StatefulPartitionedCall2H
"dropout_37/StatefulPartitionedCall"dropout_37/StatefulPartitionedCall2H
"dropout_38/StatefulPartitionedCall"dropout_38/StatefulPartitionedCall2B
lstm_21/StatefulPartitionedCalllstm_21/StatefulPartitionedCall2B
lstm_22/StatefulPartitionedCalllstm_22/StatefulPartitionedCall2B
lstm_23/StatefulPartitionedCalllstm_23/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
х
├
while_cond_681958
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_681958___redundant_placeholder04
0while_while_cond_681958___redundant_placeholder14
0while_while_cond_681958___redundant_placeholder24
0while_while_cond_681958___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
ЎJ
ў
C__inference_lstm_22_layer_call_and_return_conditional_losses_684701

inputs=
+lstm_cell_22_matmul_readvariableop_resource:@?
-lstm_cell_22_matmul_1_readvariableop_resource:@:
,lstm_cell_22_biasadd_readvariableop_resource:@
identityѕб#lstm_cell_22/BiasAdd/ReadVariableOpб"lstm_cell_22/MatMul/ReadVariableOpб$lstm_cell_22/MatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskј
"lstm_cell_22/MatMul/ReadVariableOpReadVariableOp+lstm_cell_22_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ћ
lstm_cell_22/MatMulMatMulstrided_slice_2:output:0*lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @њ
$lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_22_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell_22/MatMul_1MatMulzeros:output:0,lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @І
lstm_cell_22/addAddV2lstm_cell_22/MatMul:product:0lstm_cell_22/MatMul_1:product:0*
T0*'
_output_shapes
:         @ї
#lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ћ
lstm_cell_22/BiasAddBiasAddlstm_cell_22/add:z:0+lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @^
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0lstm_cell_22/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitn
lstm_cell_22/SigmoidSigmoidlstm_cell_22/split:output:0*
T0*'
_output_shapes
:         p
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/split:output:1*
T0*'
_output_shapes
:         w
lstm_cell_22/mulMullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         h
lstm_cell_22/ReluRelulstm_cell_22/split:output:2*
T0*'
_output_shapes
:         є
lstm_cell_22/mul_1Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Relu:activations:0*
T0*'
_output_shapes
:         {
lstm_cell_22/add_1AddV2lstm_cell_22/mul:z:0lstm_cell_22/mul_1:z:0*
T0*'
_output_shapes
:         p
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/split:output:3*
T0*'
_output_shapes
:         e
lstm_cell_22/Relu_1Relulstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         і
lstm_cell_22/mul_2Mullstm_cell_22/Sigmoid_2:y:0!lstm_cell_22/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѓ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_22_matmul_readvariableop_resource-lstm_cell_22_matmul_1_readvariableop_resource,lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_684617*
condR
while_cond_684616*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         └
NoOpNoOp$^lstm_cell_22/BiasAdd/ReadVariableOp#^lstm_cell_22/MatMul/ReadVariableOp%^lstm_cell_22/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2J
#lstm_cell_22/BiasAdd/ReadVariableOp#lstm_cell_22/BiasAdd/ReadVariableOp2H
"lstm_cell_22/MatMul/ReadVariableOp"lstm_cell_22/MatMul/ReadVariableOp2L
$lstm_cell_22/MatMul_1/ReadVariableOp$lstm_cell_22/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
х
├
while_cond_680343
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_680343___redundant_placeholder04
0while_while_cond_680343___redundant_placeholder14
0while_while_cond_680343___redundant_placeholder24
0while_while_cond_680343___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
с"
П
while_body_680535
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_22_680559_0:@-
while_lstm_cell_22_680561_0:@)
while_lstm_cell_22_680563_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_22_680559:@+
while_lstm_cell_22_680561:@'
while_lstm_cell_22_680563:@ѕб*while/lstm_cell_22/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Х
*while/lstm_cell_22/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_22_680559_0while_lstm_cell_22_680561_0while_lstm_cell_22_680563_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_680476▄
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_22/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: љ
while/Identity_4Identity3while/lstm_cell_22/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         љ
while/Identity_5Identity3while/lstm_cell_22/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         y

while/NoOpNoOp+^while/lstm_cell_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_22_680559while_lstm_cell_22_680559_0"8
while_lstm_cell_22_680561while_lstm_cell_22_680561_0"8
while_lstm_cell_22_680563while_lstm_cell_22_680563_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2X
*while/lstm_cell_22/StatefulPartitionedCall*while/lstm_cell_22/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
М
Ѓ
H__inference_lstm_cell_23_layer_call_and_return_conditional_losses_685936

inputs
states_0
states_10
matmul_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         Љ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
states/0:QM
'
_output_shapes
:         
"
_user_specified_name
states/1
х
├
while_cond_685259
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_685259___redundant_placeholder04
0while_while_cond_685259___redundant_placeholder14
0while_while_cond_685259___redundant_placeholder24
0while_while_cond_685259___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
ЯP
в
&sequential_7_lstm_21_while_body_679518F
Bsequential_7_lstm_21_while_sequential_7_lstm_21_while_loop_counterL
Hsequential_7_lstm_21_while_sequential_7_lstm_21_while_maximum_iterations*
&sequential_7_lstm_21_while_placeholder,
(sequential_7_lstm_21_while_placeholder_1,
(sequential_7_lstm_21_while_placeholder_2,
(sequential_7_lstm_21_while_placeholder_3E
Asequential_7_lstm_21_while_sequential_7_lstm_21_strided_slice_1_0Ђ
}sequential_7_lstm_21_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_21_tensorarrayunstack_tensorlistfromtensor_0Z
Hsequential_7_lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0:@\
Jsequential_7_lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0:@W
Isequential_7_lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0:@'
#sequential_7_lstm_21_while_identity)
%sequential_7_lstm_21_while_identity_1)
%sequential_7_lstm_21_while_identity_2)
%sequential_7_lstm_21_while_identity_3)
%sequential_7_lstm_21_while_identity_4)
%sequential_7_lstm_21_while_identity_5C
?sequential_7_lstm_21_while_sequential_7_lstm_21_strided_slice_1
{sequential_7_lstm_21_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_21_tensorarrayunstack_tensorlistfromtensorX
Fsequential_7_lstm_21_while_lstm_cell_21_matmul_readvariableop_resource:@Z
Hsequential_7_lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource:@U
Gsequential_7_lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource:@ѕб>sequential_7/lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOpб=sequential_7/lstm_21/while/lstm_cell_21/MatMul/ReadVariableOpб?sequential_7/lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpЮ
Lsequential_7/lstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Ј
>sequential_7/lstm_21/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_7_lstm_21_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_21_tensorarrayunstack_tensorlistfromtensor_0&sequential_7_lstm_21_while_placeholderUsequential_7/lstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0к
=sequential_7/lstm_21/while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOpHsequential_7_lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0Э
.sequential_7/lstm_21/while/lstm_cell_21/MatMulMatMulEsequential_7/lstm_21/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_7/lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @╩
?sequential_7/lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOpJsequential_7_lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0▀
0sequential_7/lstm_21/while/lstm_cell_21/MatMul_1MatMul(sequential_7_lstm_21_while_placeholder_2Gsequential_7/lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @▄
+sequential_7/lstm_21/while/lstm_cell_21/addAddV28sequential_7/lstm_21/while/lstm_cell_21/MatMul:product:0:sequential_7/lstm_21/while/lstm_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:         @─
>sequential_7/lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOpIsequential_7_lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0т
/sequential_7/lstm_21/while/lstm_cell_21/BiasAddBiasAdd/sequential_7/lstm_21/while/lstm_cell_21/add:z:0Fsequential_7/lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @y
7sequential_7/lstm_21/while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :«
-sequential_7/lstm_21/while/lstm_cell_21/splitSplit@sequential_7/lstm_21/while/lstm_cell_21/split/split_dim:output:08sequential_7/lstm_21/while/lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitц
/sequential_7/lstm_21/while/lstm_cell_21/SigmoidSigmoid6sequential_7/lstm_21/while/lstm_cell_21/split:output:0*
T0*'
_output_shapes
:         д
1sequential_7/lstm_21/while/lstm_cell_21/Sigmoid_1Sigmoid6sequential_7/lstm_21/while/lstm_cell_21/split:output:1*
T0*'
_output_shapes
:         ┼
+sequential_7/lstm_21/while/lstm_cell_21/mulMul5sequential_7/lstm_21/while/lstm_cell_21/Sigmoid_1:y:0(sequential_7_lstm_21_while_placeholder_3*
T0*'
_output_shapes
:         ъ
,sequential_7/lstm_21/while/lstm_cell_21/ReluRelu6sequential_7/lstm_21/while/lstm_cell_21/split:output:2*
T0*'
_output_shapes
:         О
-sequential_7/lstm_21/while/lstm_cell_21/mul_1Mul3sequential_7/lstm_21/while/lstm_cell_21/Sigmoid:y:0:sequential_7/lstm_21/while/lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         ╠
-sequential_7/lstm_21/while/lstm_cell_21/add_1AddV2/sequential_7/lstm_21/while/lstm_cell_21/mul:z:01sequential_7/lstm_21/while/lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         д
1sequential_7/lstm_21/while/lstm_cell_21/Sigmoid_2Sigmoid6sequential_7/lstm_21/while/lstm_cell_21/split:output:3*
T0*'
_output_shapes
:         Џ
.sequential_7/lstm_21/while/lstm_cell_21/Relu_1Relu1sequential_7/lstm_21/while/lstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         █
-sequential_7/lstm_21/while/lstm_cell_21/mul_2Mul5sequential_7/lstm_21/while/lstm_cell_21/Sigmoid_2:y:0<sequential_7/lstm_21/while/lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         Ў
?sequential_7/lstm_21/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_7_lstm_21_while_placeholder_1&sequential_7_lstm_21_while_placeholder1sequential_7/lstm_21/while/lstm_cell_21/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмb
 sequential_7/lstm_21/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Џ
sequential_7/lstm_21/while/addAddV2&sequential_7_lstm_21_while_placeholder)sequential_7/lstm_21/while/add/y:output:0*
T0*
_output_shapes
: d
"sequential_7/lstm_21/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :╗
 sequential_7/lstm_21/while/add_1AddV2Bsequential_7_lstm_21_while_sequential_7_lstm_21_while_loop_counter+sequential_7/lstm_21/while/add_1/y:output:0*
T0*
_output_shapes
: ў
#sequential_7/lstm_21/while/IdentityIdentity$sequential_7/lstm_21/while/add_1:z:0 ^sequential_7/lstm_21/while/NoOp*
T0*
_output_shapes
: Й
%sequential_7/lstm_21/while/Identity_1IdentityHsequential_7_lstm_21_while_sequential_7_lstm_21_while_maximum_iterations ^sequential_7/lstm_21/while/NoOp*
T0*
_output_shapes
: ў
%sequential_7/lstm_21/while/Identity_2Identity"sequential_7/lstm_21/while/add:z:0 ^sequential_7/lstm_21/while/NoOp*
T0*
_output_shapes
: ┼
%sequential_7/lstm_21/while/Identity_3IdentityOsequential_7/lstm_21/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_7/lstm_21/while/NoOp*
T0*
_output_shapes
: И
%sequential_7/lstm_21/while/Identity_4Identity1sequential_7/lstm_21/while/lstm_cell_21/mul_2:z:0 ^sequential_7/lstm_21/while/NoOp*
T0*'
_output_shapes
:         И
%sequential_7/lstm_21/while/Identity_5Identity1sequential_7/lstm_21/while/lstm_cell_21/add_1:z:0 ^sequential_7/lstm_21/while/NoOp*
T0*'
_output_shapes
:         ц
sequential_7/lstm_21/while/NoOpNoOp?^sequential_7/lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp>^sequential_7/lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp@^sequential_7/lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "S
#sequential_7_lstm_21_while_identity,sequential_7/lstm_21/while/Identity:output:0"W
%sequential_7_lstm_21_while_identity_1.sequential_7/lstm_21/while/Identity_1:output:0"W
%sequential_7_lstm_21_while_identity_2.sequential_7/lstm_21/while/Identity_2:output:0"W
%sequential_7_lstm_21_while_identity_3.sequential_7/lstm_21/while/Identity_3:output:0"W
%sequential_7_lstm_21_while_identity_4.sequential_7/lstm_21/while/Identity_4:output:0"W
%sequential_7_lstm_21_while_identity_5.sequential_7/lstm_21/while/Identity_5:output:0"ћ
Gsequential_7_lstm_21_while_lstm_cell_21_biasadd_readvariableop_resourceIsequential_7_lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0"ќ
Hsequential_7_lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resourceJsequential_7_lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0"њ
Fsequential_7_lstm_21_while_lstm_cell_21_matmul_readvariableop_resourceHsequential_7_lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0"ё
?sequential_7_lstm_21_while_sequential_7_lstm_21_strided_slice_1Asequential_7_lstm_21_while_sequential_7_lstm_21_strided_slice_1_0"Ч
{sequential_7_lstm_21_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_21_tensorarrayunstack_tensorlistfromtensor}sequential_7_lstm_21_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_21_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2ђ
>sequential_7/lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp>sequential_7/lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp2~
=sequential_7/lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp=sequential_7/lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp2ѓ
?sequential_7/lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp?sequential_7/lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
 7
╩
while_body_685260
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_23_matmul_readvariableop_resource_0:@G
5while_lstm_cell_23_matmul_1_readvariableop_resource_0:@B
4while_lstm_cell_23_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_23_matmul_readvariableop_resource:@E
3while_lstm_cell_23_matmul_1_readvariableop_resource:@@
2while_lstm_cell_23_biasadd_readvariableop_resource:@ѕб)while/lstm_cell_23/BiasAdd/ReadVariableOpб(while/lstm_cell_23/MatMul/ReadVariableOpб*while/lstm_cell_23/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ю
(while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_23_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0╣
while/lstm_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @а
*while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_23_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0а
while/lstm_cell_23/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_23/addAddV2#while/lstm_cell_23/MatMul:product:0%while/lstm_cell_23/MatMul_1:product:0*
T0*'
_output_shapes
:         @џ
)while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0д
while/lstm_cell_23/BiasAddBiasAddwhile/lstm_cell_23/add:z:01while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
"while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_23/splitSplit+while/lstm_cell_23/split/split_dim:output:0#while/lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitz
while/lstm_cell_23/SigmoidSigmoid!while/lstm_cell_23/split:output:0*
T0*'
_output_shapes
:         |
while/lstm_cell_23/Sigmoid_1Sigmoid!while/lstm_cell_23/split:output:1*
T0*'
_output_shapes
:         є
while/lstm_cell_23/mulMul while/lstm_cell_23/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         t
while/lstm_cell_23/ReluRelu!while/lstm_cell_23/split:output:2*
T0*'
_output_shapes
:         ў
while/lstm_cell_23/mul_1Mulwhile/lstm_cell_23/Sigmoid:y:0%while/lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         Ї
while/lstm_cell_23/add_1AddV2while/lstm_cell_23/mul:z:0while/lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         |
while/lstm_cell_23/Sigmoid_2Sigmoid!while/lstm_cell_23/split:output:3*
T0*'
_output_shapes
:         q
while/lstm_cell_23/Relu_1Reluwhile/lstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         ю
while/lstm_cell_23/mul_2Mul while/lstm_cell_23/Sigmoid_2:y:0'while/lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_23/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_23/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         y
while/Identity_5Identitywhile/lstm_cell_23/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         л

while/NoOpNoOp*^while/lstm_cell_23/BiasAdd/ReadVariableOp)^while/lstm_cell_23/MatMul/ReadVariableOp+^while/lstm_cell_23/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_23_biasadd_readvariableop_resource4while_lstm_cell_23_biasadd_readvariableop_resource_0"l
3while_lstm_cell_23_matmul_1_readvariableop_resource5while_lstm_cell_23_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_23_matmul_readvariableop_resource3while_lstm_cell_23_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2V
)while/lstm_cell_23/BiasAdd/ReadVariableOp)while/lstm_cell_23/BiasAdd/ReadVariableOp2T
(while/lstm_cell_23/MatMul/ReadVariableOp(while/lstm_cell_23/MatMul/ReadVariableOp2X
*while/lstm_cell_23/MatMul_1/ReadVariableOp*while/lstm_cell_23/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
 
▓
(__inference_lstm_21_layer_call_fn_683618

inputs
unknown:@
	unknown_0:@
	unknown_1:@
identityѕбStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_lstm_21_layer_call_and_return_conditional_losses_681112s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ъJ
ў
C__inference_lstm_23_layer_call_and_return_conditional_losses_681426

inputs=
+lstm_cell_23_matmul_readvariableop_resource:@?
-lstm_cell_23_matmul_1_readvariableop_resource:@:
,lstm_cell_23_biasadd_readvariableop_resource:@
identityѕб#lstm_cell_23/BiasAdd/ReadVariableOpб"lstm_cell_23/MatMul/ReadVariableOpб$lstm_cell_23/MatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskј
"lstm_cell_23/MatMul/ReadVariableOpReadVariableOp+lstm_cell_23_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ћ
lstm_cell_23/MatMulMatMulstrided_slice_2:output:0*lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @њ
$lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_23_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell_23/MatMul_1MatMulzeros:output:0,lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @І
lstm_cell_23/addAddV2lstm_cell_23/MatMul:product:0lstm_cell_23/MatMul_1:product:0*
T0*'
_output_shapes
:         @ї
#lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ћ
lstm_cell_23/BiasAddBiasAddlstm_cell_23/add:z:0+lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @^
lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_23/splitSplit%lstm_cell_23/split/split_dim:output:0lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitn
lstm_cell_23/SigmoidSigmoidlstm_cell_23/split:output:0*
T0*'
_output_shapes
:         p
lstm_cell_23/Sigmoid_1Sigmoidlstm_cell_23/split:output:1*
T0*'
_output_shapes
:         w
lstm_cell_23/mulMullstm_cell_23/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         h
lstm_cell_23/ReluRelulstm_cell_23/split:output:2*
T0*'
_output_shapes
:         є
lstm_cell_23/mul_1Mullstm_cell_23/Sigmoid:y:0lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         {
lstm_cell_23/add_1AddV2lstm_cell_23/mul:z:0lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         p
lstm_cell_23/Sigmoid_2Sigmoidlstm_cell_23/split:output:3*
T0*'
_output_shapes
:         e
lstm_cell_23/Relu_1Relulstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         і
lstm_cell_23/mul_2Mullstm_cell_23/Sigmoid_2:y:0!lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѓ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_23_matmul_readvariableop_resource-lstm_cell_23_matmul_1_readvariableop_resource,lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_681342*
condR
while_cond_681341*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         └
NoOpNoOp$^lstm_cell_23/BiasAdd/ReadVariableOp#^lstm_cell_23/MatMul/ReadVariableOp%^lstm_cell_23/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2J
#lstm_cell_23/BiasAdd/ReadVariableOp#lstm_cell_23/BiasAdd/ReadVariableOp2H
"lstm_cell_23/MatMul/ReadVariableOp"lstm_cell_23/MatMul/ReadVariableOp2L
$lstm_cell_23/MatMul_1/ReadVariableOp$lstm_cell_23/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
М
Ѓ
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_685870

inputs
states_0
states_10
matmul_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         Љ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
states/0:QM
'
_output_shapes
:         
"
_user_specified_name
states/1
х
├
while_cond_680884
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_680884___redundant_placeholder04
0while_while_cond_680884___redundant_placeholder14
0while_while_cond_680884___redundant_placeholder24
0while_while_cond_680884___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
»ы
¤'
"__inference__traced_restore_686361
file_prefix2
 assignvariableop_dense_19_kernel:.
 assignvariableop_1_dense_19_bias:4
"assignvariableop_2_dense_20_kernel:.
 assignvariableop_3_dense_20_bias:4
"assignvariableop_4_dense_21_kernel:.
 assignvariableop_5_dense_21_bias:4
"assignvariableop_6_dense_22_kernel:.
 assignvariableop_7_dense_22_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: A
/assignvariableop_13_lstm_21_lstm_cell_21_kernel:@K
9assignvariableop_14_lstm_21_lstm_cell_21_recurrent_kernel:@;
-assignvariableop_15_lstm_21_lstm_cell_21_bias:@A
/assignvariableop_16_lstm_22_lstm_cell_22_kernel:@K
9assignvariableop_17_lstm_22_lstm_cell_22_recurrent_kernel:@;
-assignvariableop_18_lstm_22_lstm_cell_22_bias:@A
/assignvariableop_19_lstm_23_lstm_cell_23_kernel:@K
9assignvariableop_20_lstm_23_lstm_cell_23_recurrent_kernel:@;
-assignvariableop_21_lstm_23_lstm_cell_23_bias:@#
assignvariableop_22_total: #
assignvariableop_23_count: %
assignvariableop_24_total_1: %
assignvariableop_25_count_1: <
*assignvariableop_26_adam_dense_19_kernel_m:6
(assignvariableop_27_adam_dense_19_bias_m:<
*assignvariableop_28_adam_dense_20_kernel_m:6
(assignvariableop_29_adam_dense_20_bias_m:<
*assignvariableop_30_adam_dense_21_kernel_m:6
(assignvariableop_31_adam_dense_21_bias_m:<
*assignvariableop_32_adam_dense_22_kernel_m:6
(assignvariableop_33_adam_dense_22_bias_m:H
6assignvariableop_34_adam_lstm_21_lstm_cell_21_kernel_m:@R
@assignvariableop_35_adam_lstm_21_lstm_cell_21_recurrent_kernel_m:@B
4assignvariableop_36_adam_lstm_21_lstm_cell_21_bias_m:@H
6assignvariableop_37_adam_lstm_22_lstm_cell_22_kernel_m:@R
@assignvariableop_38_adam_lstm_22_lstm_cell_22_recurrent_kernel_m:@B
4assignvariableop_39_adam_lstm_22_lstm_cell_22_bias_m:@H
6assignvariableop_40_adam_lstm_23_lstm_cell_23_kernel_m:@R
@assignvariableop_41_adam_lstm_23_lstm_cell_23_recurrent_kernel_m:@B
4assignvariableop_42_adam_lstm_23_lstm_cell_23_bias_m:@<
*assignvariableop_43_adam_dense_19_kernel_v:6
(assignvariableop_44_adam_dense_19_bias_v:<
*assignvariableop_45_adam_dense_20_kernel_v:6
(assignvariableop_46_adam_dense_20_bias_v:<
*assignvariableop_47_adam_dense_21_kernel_v:6
(assignvariableop_48_adam_dense_21_bias_v:<
*assignvariableop_49_adam_dense_22_kernel_v:6
(assignvariableop_50_adam_dense_22_bias_v:H
6assignvariableop_51_adam_lstm_21_lstm_cell_21_kernel_v:@R
@assignvariableop_52_adam_lstm_21_lstm_cell_21_recurrent_kernel_v:@B
4assignvariableop_53_adam_lstm_21_lstm_cell_21_bias_v:@H
6assignvariableop_54_adam_lstm_22_lstm_cell_22_kernel_v:@R
@assignvariableop_55_adam_lstm_22_lstm_cell_22_recurrent_kernel_v:@B
4assignvariableop_56_adam_lstm_22_lstm_cell_22_bias_v:@H
6assignvariableop_57_adam_lstm_23_lstm_cell_23_kernel_v:@R
@assignvariableop_58_adam_lstm_23_lstm_cell_23_recurrent_kernel_v:@B
4assignvariableop_59_adam_lstm_23_lstm_cell_23_bias_v:@
identity_61ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_47бAssignVariableOp_48бAssignVariableOp_49бAssignVariableOp_5бAssignVariableOp_50бAssignVariableOp_51бAssignVariableOp_52бAssignVariableOp_53бAssignVariableOp_54бAssignVariableOp_55бAssignVariableOp_56бAssignVariableOp_57бAssignVariableOp_58бAssignVariableOp_59бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9Р
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*ѕ
value■Bч=B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHь
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*Ј
valueЁBѓ=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B м
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*і
_output_shapesэ
З:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*K
dtypesA
?2=	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOpAssignVariableOp assignvariableop_dense_19_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_19_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_20_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_20_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_21_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_21_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_22_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_22_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:І
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_13AssignVariableOp/assignvariableop_13_lstm_21_lstm_cell_21_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_14AssignVariableOp9assignvariableop_14_lstm_21_lstm_cell_21_recurrent_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:ъ
AssignVariableOp_15AssignVariableOp-assignvariableop_15_lstm_21_lstm_cell_21_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_16AssignVariableOp/assignvariableop_16_lstm_22_lstm_cell_22_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_17AssignVariableOp9assignvariableop_17_lstm_22_lstm_cell_22_recurrent_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:ъ
AssignVariableOp_18AssignVariableOp-assignvariableop_18_lstm_22_lstm_cell_22_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_19AssignVariableOp/assignvariableop_19_lstm_23_lstm_cell_23_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_20AssignVariableOp9assignvariableop_20_lstm_23_lstm_cell_23_recurrent_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:ъ
AssignVariableOp_21AssignVariableOp-assignvariableop_21_lstm_23_lstm_cell_23_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_24AssignVariableOpassignvariableop_24_total_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_25AssignVariableOpassignvariableop_25_count_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_dense_19_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_dense_19_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_dense_20_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_dense_20_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_dense_21_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_dense_21_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_dense_22_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_dense_22_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_34AssignVariableOp6assignvariableop_34_adam_lstm_21_lstm_cell_21_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOp_35AssignVariableOp@assignvariableop_35_adam_lstm_21_lstm_cell_21_recurrent_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_36AssignVariableOp4assignvariableop_36_adam_lstm_21_lstm_cell_21_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_37AssignVariableOp6assignvariableop_37_adam_lstm_22_lstm_cell_22_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOp_38AssignVariableOp@assignvariableop_38_adam_lstm_22_lstm_cell_22_recurrent_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_39AssignVariableOp4assignvariableop_39_adam_lstm_22_lstm_cell_22_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_40AssignVariableOp6assignvariableop_40_adam_lstm_23_lstm_cell_23_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOp_41AssignVariableOp@assignvariableop_41_adam_lstm_23_lstm_cell_23_recurrent_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_42AssignVariableOp4assignvariableop_42_adam_lstm_23_lstm_cell_23_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_19_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_19_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_dense_20_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_dense_20_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_dense_21_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_dense_21_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_dense_22_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_dense_22_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_51AssignVariableOp6assignvariableop_51_adam_lstm_21_lstm_cell_21_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOp_52AssignVariableOp@assignvariableop_52_adam_lstm_21_lstm_cell_21_recurrent_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_53AssignVariableOp4assignvariableop_53_adam_lstm_21_lstm_cell_21_bias_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_54AssignVariableOp6assignvariableop_54_adam_lstm_22_lstm_cell_22_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOp_55AssignVariableOp@assignvariableop_55_adam_lstm_22_lstm_cell_22_recurrent_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_56AssignVariableOp4assignvariableop_56_adam_lstm_22_lstm_cell_22_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_57AssignVariableOp6assignvariableop_57_adam_lstm_23_lstm_cell_23_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOp_58AssignVariableOp@assignvariableop_58_adam_lstm_23_lstm_cell_23_recurrent_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_59AssignVariableOp4assignvariableop_59_adam_lstm_23_lstm_cell_23_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 э

Identity_60Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_61IdentityIdentity_60:output:0^NoOp_1*
T0*
_output_shapes
: С

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_61Identity_61:output:0*Ї
_input_shapes|
z: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
х
├
while_cond_684973
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_684973___redundant_placeholder04
0while_while_cond_684973___redundant_placeholder14
0while_while_cond_684973___redundant_placeholder24
0while_while_cond_684973___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
 7
╩
while_body_681342
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_23_matmul_readvariableop_resource_0:@G
5while_lstm_cell_23_matmul_1_readvariableop_resource_0:@B
4while_lstm_cell_23_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_23_matmul_readvariableop_resource:@E
3while_lstm_cell_23_matmul_1_readvariableop_resource:@@
2while_lstm_cell_23_biasadd_readvariableop_resource:@ѕб)while/lstm_cell_23/BiasAdd/ReadVariableOpб(while/lstm_cell_23/MatMul/ReadVariableOpб*while/lstm_cell_23/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ю
(while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_23_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0╣
while/lstm_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @а
*while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_23_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0а
while/lstm_cell_23/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_23/addAddV2#while/lstm_cell_23/MatMul:product:0%while/lstm_cell_23/MatMul_1:product:0*
T0*'
_output_shapes
:         @џ
)while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0д
while/lstm_cell_23/BiasAddBiasAddwhile/lstm_cell_23/add:z:01while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
"while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_23/splitSplit+while/lstm_cell_23/split/split_dim:output:0#while/lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitz
while/lstm_cell_23/SigmoidSigmoid!while/lstm_cell_23/split:output:0*
T0*'
_output_shapes
:         |
while/lstm_cell_23/Sigmoid_1Sigmoid!while/lstm_cell_23/split:output:1*
T0*'
_output_shapes
:         є
while/lstm_cell_23/mulMul while/lstm_cell_23/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         t
while/lstm_cell_23/ReluRelu!while/lstm_cell_23/split:output:2*
T0*'
_output_shapes
:         ў
while/lstm_cell_23/mul_1Mulwhile/lstm_cell_23/Sigmoid:y:0%while/lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         Ї
while/lstm_cell_23/add_1AddV2while/lstm_cell_23/mul:z:0while/lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         |
while/lstm_cell_23/Sigmoid_2Sigmoid!while/lstm_cell_23/split:output:3*
T0*'
_output_shapes
:         q
while/lstm_cell_23/Relu_1Reluwhile/lstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         ю
while/lstm_cell_23/mul_2Mul while/lstm_cell_23/Sigmoid_2:y:0'while/lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_23/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_23/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         y
while/Identity_5Identitywhile/lstm_cell_23/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         л

while/NoOpNoOp*^while/lstm_cell_23/BiasAdd/ReadVariableOp)^while/lstm_cell_23/MatMul/ReadVariableOp+^while/lstm_cell_23/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_23_biasadd_readvariableop_resource4while_lstm_cell_23_biasadd_readvariableop_resource_0"l
3while_lstm_cell_23_matmul_1_readvariableop_resource5while_lstm_cell_23_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_23_matmul_readvariableop_resource3while_lstm_cell_23_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2V
)while/lstm_cell_23/BiasAdd/ReadVariableOp)while/lstm_cell_23/BiasAdd/ReadVariableOp2T
(while/lstm_cell_23/MatMul/ReadVariableOp(while/lstm_cell_23/MatMul/ReadVariableOp2X
*while/lstm_cell_23/MatMul_1/ReadVariableOp*while/lstm_cell_23/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
╦
Ђ
H__inference_lstm_cell_23_layer_call_and_return_conditional_losses_680826

inputs

states
states_10
matmul_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         Љ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_namestates:OK
'
_output_shapes
:         
 
_user_specified_namestates
З	
e
F__inference_dropout_37_layer_call_and_return_conditional_losses_681630

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
в
з
-__inference_lstm_cell_23_layer_call_fn_685887

inputs
states_0
states_1
unknown:@
	unknown_0:@
	unknown_1:@
identity

identity_1

identity_2ѕбStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_lstm_cell_23_layer_call_and_return_conditional_losses_680680o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
states/0:QM
'
_output_shapes
:         
"
_user_specified_name
states/1
с"
П
while_body_679994
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_21_680018_0:@-
while_lstm_cell_21_680020_0:@)
while_lstm_cell_21_680022_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_21_680018:@+
while_lstm_cell_21_680020:@'
while_lstm_cell_21_680022:@ѕб*while/lstm_cell_21/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Х
*while/lstm_cell_21/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_21_680018_0while_lstm_cell_21_680020_0while_lstm_cell_21_680022_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_lstm_cell_21_layer_call_and_return_conditional_losses_679980▄
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_21/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: љ
while/Identity_4Identity3while/lstm_cell_21/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         љ
while/Identity_5Identity3while/lstm_cell_21/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         y

while/NoOpNoOp+^while/lstm_cell_21/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_21_680018while_lstm_cell_21_680018_0"8
while_lstm_cell_21_680020while_lstm_cell_21_680020_0"8
while_lstm_cell_21_680022while_lstm_cell_21_680022_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2X
*while/lstm_cell_21/StatefulPartitionedCall*while/lstm_cell_21/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
ИA
╩

lstm_21_while_body_682694,
(lstm_21_while_lstm_21_while_loop_counter2
.lstm_21_while_lstm_21_while_maximum_iterations
lstm_21_while_placeholder
lstm_21_while_placeholder_1
lstm_21_while_placeholder_2
lstm_21_while_placeholder_3+
'lstm_21_while_lstm_21_strided_slice_1_0g
clstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensor_0M
;lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0:@O
=lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0:@J
<lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0:@
lstm_21_while_identity
lstm_21_while_identity_1
lstm_21_while_identity_2
lstm_21_while_identity_3
lstm_21_while_identity_4
lstm_21_while_identity_5)
%lstm_21_while_lstm_21_strided_slice_1e
alstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensorK
9lstm_21_while_lstm_cell_21_matmul_readvariableop_resource:@M
;lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource:@H
:lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource:@ѕб1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOpб0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOpб2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpљ
?lstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╬
1lstm_21/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensor_0lstm_21_while_placeholderHlstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0г
0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp;lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0Л
!lstm_21/while/lstm_cell_21/MatMulMatMul8lstm_21/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @░
2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp=lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0И
#lstm_21/while/lstm_cell_21/MatMul_1MatMullstm_21_while_placeholder_2:lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @х
lstm_21/while/lstm_cell_21/addAddV2+lstm_21/while/lstm_cell_21/MatMul:product:0-lstm_21/while/lstm_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:         @ф
1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp<lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Й
"lstm_21/while/lstm_cell_21/BiasAddBiasAdd"lstm_21/while/lstm_cell_21/add:z:09lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @l
*lstm_21/while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Є
 lstm_21/while/lstm_cell_21/splitSplit3lstm_21/while/lstm_cell_21/split/split_dim:output:0+lstm_21/while/lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitі
"lstm_21/while/lstm_cell_21/SigmoidSigmoid)lstm_21/while/lstm_cell_21/split:output:0*
T0*'
_output_shapes
:         ї
$lstm_21/while/lstm_cell_21/Sigmoid_1Sigmoid)lstm_21/while/lstm_cell_21/split:output:1*
T0*'
_output_shapes
:         ъ
lstm_21/while/lstm_cell_21/mulMul(lstm_21/while/lstm_cell_21/Sigmoid_1:y:0lstm_21_while_placeholder_3*
T0*'
_output_shapes
:         ё
lstm_21/while/lstm_cell_21/ReluRelu)lstm_21/while/lstm_cell_21/split:output:2*
T0*'
_output_shapes
:         ░
 lstm_21/while/lstm_cell_21/mul_1Mul&lstm_21/while/lstm_cell_21/Sigmoid:y:0-lstm_21/while/lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         Ц
 lstm_21/while/lstm_cell_21/add_1AddV2"lstm_21/while/lstm_cell_21/mul:z:0$lstm_21/while/lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         ї
$lstm_21/while/lstm_cell_21/Sigmoid_2Sigmoid)lstm_21/while/lstm_cell_21/split:output:3*
T0*'
_output_shapes
:         Ђ
!lstm_21/while/lstm_cell_21/Relu_1Relu$lstm_21/while/lstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         ┤
 lstm_21/while/lstm_cell_21/mul_2Mul(lstm_21/while/lstm_cell_21/Sigmoid_2:y:0/lstm_21/while/lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         т
2lstm_21/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_21_while_placeholder_1lstm_21_while_placeholder$lstm_21/while/lstm_cell_21/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмU
lstm_21/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_21/while/addAddV2lstm_21_while_placeholderlstm_21/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_21/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Є
lstm_21/while/add_1AddV2(lstm_21_while_lstm_21_while_loop_counterlstm_21/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_21/while/IdentityIdentitylstm_21/while/add_1:z:0^lstm_21/while/NoOp*
T0*
_output_shapes
: і
lstm_21/while/Identity_1Identity.lstm_21_while_lstm_21_while_maximum_iterations^lstm_21/while/NoOp*
T0*
_output_shapes
: q
lstm_21/while/Identity_2Identitylstm_21/while/add:z:0^lstm_21/while/NoOp*
T0*
_output_shapes
: ъ
lstm_21/while/Identity_3IdentityBlstm_21/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_21/while/NoOp*
T0*
_output_shapes
: Љ
lstm_21/while/Identity_4Identity$lstm_21/while/lstm_cell_21/mul_2:z:0^lstm_21/while/NoOp*
T0*'
_output_shapes
:         Љ
lstm_21/while/Identity_5Identity$lstm_21/while/lstm_cell_21/add_1:z:0^lstm_21/while/NoOp*
T0*'
_output_shapes
:         ­
lstm_21/while/NoOpNoOp2^lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp1^lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp3^lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_21_while_identitylstm_21/while/Identity:output:0"=
lstm_21_while_identity_1!lstm_21/while/Identity_1:output:0"=
lstm_21_while_identity_2!lstm_21/while/Identity_2:output:0"=
lstm_21_while_identity_3!lstm_21/while/Identity_3:output:0"=
lstm_21_while_identity_4!lstm_21/while/Identity_4:output:0"=
lstm_21_while_identity_5!lstm_21/while/Identity_5:output:0"P
%lstm_21_while_lstm_21_strided_slice_1'lstm_21_while_lstm_21_strided_slice_1_0"z
:lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource<lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0"|
;lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource=lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0"x
9lstm_21_while_lstm_cell_21_matmul_readvariableop_resource;lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0"╚
alstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensorclstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2f
1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp2d
0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp2h
2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
хн
╩
!__inference__wrapped_model_679913
lstm_21_inputR
@sequential_7_lstm_21_lstm_cell_21_matmul_readvariableop_resource:@T
Bsequential_7_lstm_21_lstm_cell_21_matmul_1_readvariableop_resource:@O
Asequential_7_lstm_21_lstm_cell_21_biasadd_readvariableop_resource:@R
@sequential_7_lstm_22_lstm_cell_22_matmul_readvariableop_resource:@T
Bsequential_7_lstm_22_lstm_cell_22_matmul_1_readvariableop_resource:@O
Asequential_7_lstm_22_lstm_cell_22_biasadd_readvariableop_resource:@R
@sequential_7_lstm_23_lstm_cell_23_matmul_readvariableop_resource:@T
Bsequential_7_lstm_23_lstm_cell_23_matmul_1_readvariableop_resource:@O
Asequential_7_lstm_23_lstm_cell_23_biasadd_readvariableop_resource:@F
4sequential_7_dense_19_matmul_readvariableop_resource:C
5sequential_7_dense_19_biasadd_readvariableop_resource:F
4sequential_7_dense_20_matmul_readvariableop_resource:C
5sequential_7_dense_20_biasadd_readvariableop_resource:F
4sequential_7_dense_21_matmul_readvariableop_resource:C
5sequential_7_dense_21_biasadd_readvariableop_resource:F
4sequential_7_dense_22_matmul_readvariableop_resource:C
5sequential_7_dense_22_biasadd_readvariableop_resource:
identityѕб,sequential_7/dense_19/BiasAdd/ReadVariableOpб+sequential_7/dense_19/MatMul/ReadVariableOpб,sequential_7/dense_20/BiasAdd/ReadVariableOpб+sequential_7/dense_20/MatMul/ReadVariableOpб,sequential_7/dense_21/BiasAdd/ReadVariableOpб+sequential_7/dense_21/MatMul/ReadVariableOpб,sequential_7/dense_22/BiasAdd/ReadVariableOpб+sequential_7/dense_22/MatMul/ReadVariableOpб8sequential_7/lstm_21/lstm_cell_21/BiasAdd/ReadVariableOpб7sequential_7/lstm_21/lstm_cell_21/MatMul/ReadVariableOpб9sequential_7/lstm_21/lstm_cell_21/MatMul_1/ReadVariableOpбsequential_7/lstm_21/whileб8sequential_7/lstm_22/lstm_cell_22/BiasAdd/ReadVariableOpб7sequential_7/lstm_22/lstm_cell_22/MatMul/ReadVariableOpб9sequential_7/lstm_22/lstm_cell_22/MatMul_1/ReadVariableOpбsequential_7/lstm_22/whileб8sequential_7/lstm_23/lstm_cell_23/BiasAdd/ReadVariableOpб7sequential_7/lstm_23/lstm_cell_23/MatMul/ReadVariableOpб9sequential_7/lstm_23/lstm_cell_23/MatMul_1/ReadVariableOpбsequential_7/lstm_23/whileW
sequential_7/lstm_21/ShapeShapelstm_21_input*
T0*
_output_shapes
:r
(sequential_7/lstm_21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_7/lstm_21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_7/lstm_21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"sequential_7/lstm_21/strided_sliceStridedSlice#sequential_7/lstm_21/Shape:output:01sequential_7/lstm_21/strided_slice/stack:output:03sequential_7/lstm_21/strided_slice/stack_1:output:03sequential_7/lstm_21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#sequential_7/lstm_21/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :▓
!sequential_7/lstm_21/zeros/packedPack+sequential_7/lstm_21/strided_slice:output:0,sequential_7/lstm_21/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 sequential_7/lstm_21/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ф
sequential_7/lstm_21/zerosFill*sequential_7/lstm_21/zeros/packed:output:0)sequential_7/lstm_21/zeros/Const:output:0*
T0*'
_output_shapes
:         g
%sequential_7/lstm_21/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Х
#sequential_7/lstm_21/zeros_1/packedPack+sequential_7/lstm_21/strided_slice:output:0.sequential_7/lstm_21/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:g
"sequential_7/lstm_21/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ▒
sequential_7/lstm_21/zeros_1Fill,sequential_7/lstm_21/zeros_1/packed:output:0+sequential_7/lstm_21/zeros_1/Const:output:0*
T0*'
_output_shapes
:         x
#sequential_7/lstm_21/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ъ
sequential_7/lstm_21/transpose	Transposelstm_21_input,sequential_7/lstm_21/transpose/perm:output:0*
T0*+
_output_shapes
:         n
sequential_7/lstm_21/Shape_1Shape"sequential_7/lstm_21/transpose:y:0*
T0*
_output_shapes
:t
*sequential_7/lstm_21/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_21/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_7/lstm_21/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:─
$sequential_7/lstm_21/strided_slice_1StridedSlice%sequential_7/lstm_21/Shape_1:output:03sequential_7/lstm_21/strided_slice_1/stack:output:05sequential_7/lstm_21/strided_slice_1/stack_1:output:05sequential_7/lstm_21/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0sequential_7/lstm_21/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         з
"sequential_7/lstm_21/TensorArrayV2TensorListReserve9sequential_7/lstm_21/TensorArrayV2/element_shape:output:0-sequential_7/lstm_21/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмЏ
Jsequential_7/lstm_21/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Ъ
<sequential_7/lstm_21/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_7/lstm_21/transpose:y:0Ssequential_7/lstm_21/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмt
*sequential_7/lstm_21/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_21/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_7/lstm_21/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:м
$sequential_7/lstm_21/strided_slice_2StridedSlice"sequential_7/lstm_21/transpose:y:03sequential_7/lstm_21/strided_slice_2/stack:output:05sequential_7/lstm_21/strided_slice_2/stack_1:output:05sequential_7/lstm_21/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskИ
7sequential_7/lstm_21/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp@sequential_7_lstm_21_lstm_cell_21_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0н
(sequential_7/lstm_21/lstm_cell_21/MatMulMatMul-sequential_7/lstm_21/strided_slice_2:output:0?sequential_7/lstm_21/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @╝
9sequential_7/lstm_21/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOpBsequential_7_lstm_21_lstm_cell_21_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0╬
*sequential_7/lstm_21/lstm_cell_21/MatMul_1MatMul#sequential_7/lstm_21/zeros:output:0Asequential_7/lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @╩
%sequential_7/lstm_21/lstm_cell_21/addAddV22sequential_7/lstm_21/lstm_cell_21/MatMul:product:04sequential_7/lstm_21/lstm_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:         @Х
8sequential_7/lstm_21/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOpAsequential_7_lstm_21_lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0М
)sequential_7/lstm_21/lstm_cell_21/BiasAddBiasAdd)sequential_7/lstm_21/lstm_cell_21/add:z:0@sequential_7/lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @s
1sequential_7/lstm_21/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ю
'sequential_7/lstm_21/lstm_cell_21/splitSplit:sequential_7/lstm_21/lstm_cell_21/split/split_dim:output:02sequential_7/lstm_21/lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitў
)sequential_7/lstm_21/lstm_cell_21/SigmoidSigmoid0sequential_7/lstm_21/lstm_cell_21/split:output:0*
T0*'
_output_shapes
:         џ
+sequential_7/lstm_21/lstm_cell_21/Sigmoid_1Sigmoid0sequential_7/lstm_21/lstm_cell_21/split:output:1*
T0*'
_output_shapes
:         Х
%sequential_7/lstm_21/lstm_cell_21/mulMul/sequential_7/lstm_21/lstm_cell_21/Sigmoid_1:y:0%sequential_7/lstm_21/zeros_1:output:0*
T0*'
_output_shapes
:         њ
&sequential_7/lstm_21/lstm_cell_21/ReluRelu0sequential_7/lstm_21/lstm_cell_21/split:output:2*
T0*'
_output_shapes
:         ┼
'sequential_7/lstm_21/lstm_cell_21/mul_1Mul-sequential_7/lstm_21/lstm_cell_21/Sigmoid:y:04sequential_7/lstm_21/lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         ║
'sequential_7/lstm_21/lstm_cell_21/add_1AddV2)sequential_7/lstm_21/lstm_cell_21/mul:z:0+sequential_7/lstm_21/lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         џ
+sequential_7/lstm_21/lstm_cell_21/Sigmoid_2Sigmoid0sequential_7/lstm_21/lstm_cell_21/split:output:3*
T0*'
_output_shapes
:         Ј
(sequential_7/lstm_21/lstm_cell_21/Relu_1Relu+sequential_7/lstm_21/lstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         ╔
'sequential_7/lstm_21/lstm_cell_21/mul_2Mul/sequential_7/lstm_21/lstm_cell_21/Sigmoid_2:y:06sequential_7/lstm_21/lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         Ѓ
2sequential_7/lstm_21/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       э
$sequential_7/lstm_21/TensorArrayV2_1TensorListReserve;sequential_7/lstm_21/TensorArrayV2_1/element_shape:output:0-sequential_7/lstm_21/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм[
sequential_7/lstm_21/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-sequential_7/lstm_21/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         i
'sequential_7/lstm_21/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : е
sequential_7/lstm_21/whileWhile0sequential_7/lstm_21/while/loop_counter:output:06sequential_7/lstm_21/while/maximum_iterations:output:0"sequential_7/lstm_21/time:output:0-sequential_7/lstm_21/TensorArrayV2_1:handle:0#sequential_7/lstm_21/zeros:output:0%sequential_7/lstm_21/zeros_1:output:0-sequential_7/lstm_21/strided_slice_1:output:0Lsequential_7/lstm_21/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_7_lstm_21_lstm_cell_21_matmul_readvariableop_resourceBsequential_7_lstm_21_lstm_cell_21_matmul_1_readvariableop_resourceAsequential_7_lstm_21_lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *2
body*R(
&sequential_7_lstm_21_while_body_679518*2
cond*R(
&sequential_7_lstm_21_while_cond_679517*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations ќ
Esequential_7/lstm_21/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Ђ
7sequential_7/lstm_21/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_7/lstm_21/while:output:3Nsequential_7/lstm_21/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0}
*sequential_7/lstm_21/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         v
,sequential_7/lstm_21/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_21/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
$sequential_7/lstm_21/strided_slice_3StridedSlice@sequential_7/lstm_21/TensorArrayV2Stack/TensorListStack:tensor:03sequential_7/lstm_21/strided_slice_3/stack:output:05sequential_7/lstm_21/strided_slice_3/stack_1:output:05sequential_7/lstm_21/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskz
%sequential_7/lstm_21/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Н
 sequential_7/lstm_21/transpose_1	Transpose@sequential_7/lstm_21/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_7/lstm_21/transpose_1/perm:output:0*
T0*+
_output_shapes
:         p
sequential_7/lstm_21/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ѕ
 sequential_7/dropout_33/IdentityIdentity$sequential_7/lstm_21/transpose_1:y:0*
T0*+
_output_shapes
:         s
sequential_7/lstm_22/ShapeShape)sequential_7/dropout_33/Identity:output:0*
T0*
_output_shapes
:r
(sequential_7/lstm_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_7/lstm_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_7/lstm_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"sequential_7/lstm_22/strided_sliceStridedSlice#sequential_7/lstm_22/Shape:output:01sequential_7/lstm_22/strided_slice/stack:output:03sequential_7/lstm_22/strided_slice/stack_1:output:03sequential_7/lstm_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#sequential_7/lstm_22/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :▓
!sequential_7/lstm_22/zeros/packedPack+sequential_7/lstm_22/strided_slice:output:0,sequential_7/lstm_22/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 sequential_7/lstm_22/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ф
sequential_7/lstm_22/zerosFill*sequential_7/lstm_22/zeros/packed:output:0)sequential_7/lstm_22/zeros/Const:output:0*
T0*'
_output_shapes
:         g
%sequential_7/lstm_22/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Х
#sequential_7/lstm_22/zeros_1/packedPack+sequential_7/lstm_22/strided_slice:output:0.sequential_7/lstm_22/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:g
"sequential_7/lstm_22/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ▒
sequential_7/lstm_22/zeros_1Fill,sequential_7/lstm_22/zeros_1/packed:output:0+sequential_7/lstm_22/zeros_1/Const:output:0*
T0*'
_output_shapes
:         x
#sequential_7/lstm_22/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ║
sequential_7/lstm_22/transpose	Transpose)sequential_7/dropout_33/Identity:output:0,sequential_7/lstm_22/transpose/perm:output:0*
T0*+
_output_shapes
:         n
sequential_7/lstm_22/Shape_1Shape"sequential_7/lstm_22/transpose:y:0*
T0*
_output_shapes
:t
*sequential_7/lstm_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_7/lstm_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:─
$sequential_7/lstm_22/strided_slice_1StridedSlice%sequential_7/lstm_22/Shape_1:output:03sequential_7/lstm_22/strided_slice_1/stack:output:05sequential_7/lstm_22/strided_slice_1/stack_1:output:05sequential_7/lstm_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0sequential_7/lstm_22/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         з
"sequential_7/lstm_22/TensorArrayV2TensorListReserve9sequential_7/lstm_22/TensorArrayV2/element_shape:output:0-sequential_7/lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмЏ
Jsequential_7/lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Ъ
<sequential_7/lstm_22/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_7/lstm_22/transpose:y:0Ssequential_7/lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмt
*sequential_7/lstm_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_7/lstm_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:м
$sequential_7/lstm_22/strided_slice_2StridedSlice"sequential_7/lstm_22/transpose:y:03sequential_7/lstm_22/strided_slice_2/stack:output:05sequential_7/lstm_22/strided_slice_2/stack_1:output:05sequential_7/lstm_22/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskИ
7sequential_7/lstm_22/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp@sequential_7_lstm_22_lstm_cell_22_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0н
(sequential_7/lstm_22/lstm_cell_22/MatMulMatMul-sequential_7/lstm_22/strided_slice_2:output:0?sequential_7/lstm_22/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @╝
9sequential_7/lstm_22/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOpBsequential_7_lstm_22_lstm_cell_22_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0╬
*sequential_7/lstm_22/lstm_cell_22/MatMul_1MatMul#sequential_7/lstm_22/zeros:output:0Asequential_7/lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @╩
%sequential_7/lstm_22/lstm_cell_22/addAddV22sequential_7/lstm_22/lstm_cell_22/MatMul:product:04sequential_7/lstm_22/lstm_cell_22/MatMul_1:product:0*
T0*'
_output_shapes
:         @Х
8sequential_7/lstm_22/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOpAsequential_7_lstm_22_lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0М
)sequential_7/lstm_22/lstm_cell_22/BiasAddBiasAdd)sequential_7/lstm_22/lstm_cell_22/add:z:0@sequential_7/lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @s
1sequential_7/lstm_22/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ю
'sequential_7/lstm_22/lstm_cell_22/splitSplit:sequential_7/lstm_22/lstm_cell_22/split/split_dim:output:02sequential_7/lstm_22/lstm_cell_22/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitў
)sequential_7/lstm_22/lstm_cell_22/SigmoidSigmoid0sequential_7/lstm_22/lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         џ
+sequential_7/lstm_22/lstm_cell_22/Sigmoid_1Sigmoid0sequential_7/lstm_22/lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         Х
%sequential_7/lstm_22/lstm_cell_22/mulMul/sequential_7/lstm_22/lstm_cell_22/Sigmoid_1:y:0%sequential_7/lstm_22/zeros_1:output:0*
T0*'
_output_shapes
:         њ
&sequential_7/lstm_22/lstm_cell_22/ReluRelu0sequential_7/lstm_22/lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         ┼
'sequential_7/lstm_22/lstm_cell_22/mul_1Mul-sequential_7/lstm_22/lstm_cell_22/Sigmoid:y:04sequential_7/lstm_22/lstm_cell_22/Relu:activations:0*
T0*'
_output_shapes
:         ║
'sequential_7/lstm_22/lstm_cell_22/add_1AddV2)sequential_7/lstm_22/lstm_cell_22/mul:z:0+sequential_7/lstm_22/lstm_cell_22/mul_1:z:0*
T0*'
_output_shapes
:         џ
+sequential_7/lstm_22/lstm_cell_22/Sigmoid_2Sigmoid0sequential_7/lstm_22/lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         Ј
(sequential_7/lstm_22/lstm_cell_22/Relu_1Relu+sequential_7/lstm_22/lstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         ╔
'sequential_7/lstm_22/lstm_cell_22/mul_2Mul/sequential_7/lstm_22/lstm_cell_22/Sigmoid_2:y:06sequential_7/lstm_22/lstm_cell_22/Relu_1:activations:0*
T0*'
_output_shapes
:         Ѓ
2sequential_7/lstm_22/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       э
$sequential_7/lstm_22/TensorArrayV2_1TensorListReserve;sequential_7/lstm_22/TensorArrayV2_1/element_shape:output:0-sequential_7/lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм[
sequential_7/lstm_22/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-sequential_7/lstm_22/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         i
'sequential_7/lstm_22/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : е
sequential_7/lstm_22/whileWhile0sequential_7/lstm_22/while/loop_counter:output:06sequential_7/lstm_22/while/maximum_iterations:output:0"sequential_7/lstm_22/time:output:0-sequential_7/lstm_22/TensorArrayV2_1:handle:0#sequential_7/lstm_22/zeros:output:0%sequential_7/lstm_22/zeros_1:output:0-sequential_7/lstm_22/strided_slice_1:output:0Lsequential_7/lstm_22/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_7_lstm_22_lstm_cell_22_matmul_readvariableop_resourceBsequential_7_lstm_22_lstm_cell_22_matmul_1_readvariableop_resourceAsequential_7_lstm_22_lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *2
body*R(
&sequential_7_lstm_22_while_body_679658*2
cond*R(
&sequential_7_lstm_22_while_cond_679657*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations ќ
Esequential_7/lstm_22/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Ђ
7sequential_7/lstm_22/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_7/lstm_22/while:output:3Nsequential_7/lstm_22/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0}
*sequential_7/lstm_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         v
,sequential_7/lstm_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
$sequential_7/lstm_22/strided_slice_3StridedSlice@sequential_7/lstm_22/TensorArrayV2Stack/TensorListStack:tensor:03sequential_7/lstm_22/strided_slice_3/stack:output:05sequential_7/lstm_22/strided_slice_3/stack_1:output:05sequential_7/lstm_22/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskz
%sequential_7/lstm_22/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Н
 sequential_7/lstm_22/transpose_1	Transpose@sequential_7/lstm_22/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_7/lstm_22/transpose_1/perm:output:0*
T0*+
_output_shapes
:         p
sequential_7/lstm_22/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ѕ
 sequential_7/dropout_34/IdentityIdentity$sequential_7/lstm_22/transpose_1:y:0*
T0*+
_output_shapes
:         s
sequential_7/lstm_23/ShapeShape)sequential_7/dropout_34/Identity:output:0*
T0*
_output_shapes
:r
(sequential_7/lstm_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_7/lstm_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_7/lstm_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"sequential_7/lstm_23/strided_sliceStridedSlice#sequential_7/lstm_23/Shape:output:01sequential_7/lstm_23/strided_slice/stack:output:03sequential_7/lstm_23/strided_slice/stack_1:output:03sequential_7/lstm_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#sequential_7/lstm_23/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :▓
!sequential_7/lstm_23/zeros/packedPack+sequential_7/lstm_23/strided_slice:output:0,sequential_7/lstm_23/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 sequential_7/lstm_23/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ф
sequential_7/lstm_23/zerosFill*sequential_7/lstm_23/zeros/packed:output:0)sequential_7/lstm_23/zeros/Const:output:0*
T0*'
_output_shapes
:         g
%sequential_7/lstm_23/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Х
#sequential_7/lstm_23/zeros_1/packedPack+sequential_7/lstm_23/strided_slice:output:0.sequential_7/lstm_23/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:g
"sequential_7/lstm_23/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ▒
sequential_7/lstm_23/zeros_1Fill,sequential_7/lstm_23/zeros_1/packed:output:0+sequential_7/lstm_23/zeros_1/Const:output:0*
T0*'
_output_shapes
:         x
#sequential_7/lstm_23/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ║
sequential_7/lstm_23/transpose	Transpose)sequential_7/dropout_34/Identity:output:0,sequential_7/lstm_23/transpose/perm:output:0*
T0*+
_output_shapes
:         n
sequential_7/lstm_23/Shape_1Shape"sequential_7/lstm_23/transpose:y:0*
T0*
_output_shapes
:t
*sequential_7/lstm_23/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_23/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_7/lstm_23/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:─
$sequential_7/lstm_23/strided_slice_1StridedSlice%sequential_7/lstm_23/Shape_1:output:03sequential_7/lstm_23/strided_slice_1/stack:output:05sequential_7/lstm_23/strided_slice_1/stack_1:output:05sequential_7/lstm_23/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0sequential_7/lstm_23/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         з
"sequential_7/lstm_23/TensorArrayV2TensorListReserve9sequential_7/lstm_23/TensorArrayV2/element_shape:output:0-sequential_7/lstm_23/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмЏ
Jsequential_7/lstm_23/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Ъ
<sequential_7/lstm_23/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_7/lstm_23/transpose:y:0Ssequential_7/lstm_23/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмt
*sequential_7/lstm_23/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_23/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_7/lstm_23/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:м
$sequential_7/lstm_23/strided_slice_2StridedSlice"sequential_7/lstm_23/transpose:y:03sequential_7/lstm_23/strided_slice_2/stack:output:05sequential_7/lstm_23/strided_slice_2/stack_1:output:05sequential_7/lstm_23/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskИ
7sequential_7/lstm_23/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp@sequential_7_lstm_23_lstm_cell_23_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0н
(sequential_7/lstm_23/lstm_cell_23/MatMulMatMul-sequential_7/lstm_23/strided_slice_2:output:0?sequential_7/lstm_23/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @╝
9sequential_7/lstm_23/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOpBsequential_7_lstm_23_lstm_cell_23_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0╬
*sequential_7/lstm_23/lstm_cell_23/MatMul_1MatMul#sequential_7/lstm_23/zeros:output:0Asequential_7/lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @╩
%sequential_7/lstm_23/lstm_cell_23/addAddV22sequential_7/lstm_23/lstm_cell_23/MatMul:product:04sequential_7/lstm_23/lstm_cell_23/MatMul_1:product:0*
T0*'
_output_shapes
:         @Х
8sequential_7/lstm_23/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOpAsequential_7_lstm_23_lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0М
)sequential_7/lstm_23/lstm_cell_23/BiasAddBiasAdd)sequential_7/lstm_23/lstm_cell_23/add:z:0@sequential_7/lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @s
1sequential_7/lstm_23/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ю
'sequential_7/lstm_23/lstm_cell_23/splitSplit:sequential_7/lstm_23/lstm_cell_23/split/split_dim:output:02sequential_7/lstm_23/lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitў
)sequential_7/lstm_23/lstm_cell_23/SigmoidSigmoid0sequential_7/lstm_23/lstm_cell_23/split:output:0*
T0*'
_output_shapes
:         џ
+sequential_7/lstm_23/lstm_cell_23/Sigmoid_1Sigmoid0sequential_7/lstm_23/lstm_cell_23/split:output:1*
T0*'
_output_shapes
:         Х
%sequential_7/lstm_23/lstm_cell_23/mulMul/sequential_7/lstm_23/lstm_cell_23/Sigmoid_1:y:0%sequential_7/lstm_23/zeros_1:output:0*
T0*'
_output_shapes
:         њ
&sequential_7/lstm_23/lstm_cell_23/ReluRelu0sequential_7/lstm_23/lstm_cell_23/split:output:2*
T0*'
_output_shapes
:         ┼
'sequential_7/lstm_23/lstm_cell_23/mul_1Mul-sequential_7/lstm_23/lstm_cell_23/Sigmoid:y:04sequential_7/lstm_23/lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         ║
'sequential_7/lstm_23/lstm_cell_23/add_1AddV2)sequential_7/lstm_23/lstm_cell_23/mul:z:0+sequential_7/lstm_23/lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         џ
+sequential_7/lstm_23/lstm_cell_23/Sigmoid_2Sigmoid0sequential_7/lstm_23/lstm_cell_23/split:output:3*
T0*'
_output_shapes
:         Ј
(sequential_7/lstm_23/lstm_cell_23/Relu_1Relu+sequential_7/lstm_23/lstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         ╔
'sequential_7/lstm_23/lstm_cell_23/mul_2Mul/sequential_7/lstm_23/lstm_cell_23/Sigmoid_2:y:06sequential_7/lstm_23/lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         Ѓ
2sequential_7/lstm_23/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       э
$sequential_7/lstm_23/TensorArrayV2_1TensorListReserve;sequential_7/lstm_23/TensorArrayV2_1/element_shape:output:0-sequential_7/lstm_23/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм[
sequential_7/lstm_23/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-sequential_7/lstm_23/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         i
'sequential_7/lstm_23/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : е
sequential_7/lstm_23/whileWhile0sequential_7/lstm_23/while/loop_counter:output:06sequential_7/lstm_23/while/maximum_iterations:output:0"sequential_7/lstm_23/time:output:0-sequential_7/lstm_23/TensorArrayV2_1:handle:0#sequential_7/lstm_23/zeros:output:0%sequential_7/lstm_23/zeros_1:output:0-sequential_7/lstm_23/strided_slice_1:output:0Lsequential_7/lstm_23/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_7_lstm_23_lstm_cell_23_matmul_readvariableop_resourceBsequential_7_lstm_23_lstm_cell_23_matmul_1_readvariableop_resourceAsequential_7_lstm_23_lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *2
body*R(
&sequential_7_lstm_23_while_body_679798*2
cond*R(
&sequential_7_lstm_23_while_cond_679797*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations ќ
Esequential_7/lstm_23/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Ђ
7sequential_7/lstm_23/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_7/lstm_23/while:output:3Nsequential_7/lstm_23/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0}
*sequential_7/lstm_23/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         v
,sequential_7/lstm_23/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,sequential_7/lstm_23/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
$sequential_7/lstm_23/strided_slice_3StridedSlice@sequential_7/lstm_23/TensorArrayV2Stack/TensorListStack:tensor:03sequential_7/lstm_23/strided_slice_3/stack:output:05sequential_7/lstm_23/strided_slice_3/stack_1:output:05sequential_7/lstm_23/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskz
%sequential_7/lstm_23/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Н
 sequential_7/lstm_23/transpose_1	Transpose@sequential_7/lstm_23/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_7/lstm_23/transpose_1/perm:output:0*
T0*+
_output_shapes
:         p
sequential_7/lstm_23/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Ї
 sequential_7/dropout_35/IdentityIdentity-sequential_7/lstm_23/strided_slice_3:output:0*
T0*'
_output_shapes
:         а
+sequential_7/dense_19/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_19_matmul_readvariableop_resource*
_output_shapes

:*
dtype0И
sequential_7/dense_19/MatMulMatMul)sequential_7/dropout_35/Identity:output:03sequential_7/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ъ
,sequential_7/dense_19/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0И
sequential_7/dense_19/BiasAddBiasAdd&sequential_7/dense_19/MatMul:product:04sequential_7/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         |
sequential_7/dense_19/ReluRelu&sequential_7/dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:         ѕ
 sequential_7/dropout_36/IdentityIdentity(sequential_7/dense_19/Relu:activations:0*
T0*'
_output_shapes
:         а
+sequential_7/dense_20/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_20_matmul_readvariableop_resource*
_output_shapes

:*
dtype0И
sequential_7/dense_20/MatMulMatMul)sequential_7/dropout_36/Identity:output:03sequential_7/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ъ
,sequential_7/dense_20/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0И
sequential_7/dense_20/BiasAddBiasAdd&sequential_7/dense_20/MatMul:product:04sequential_7/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         |
sequential_7/dense_20/ReluRelu&sequential_7/dense_20/BiasAdd:output:0*
T0*'
_output_shapes
:         ѕ
 sequential_7/dropout_37/IdentityIdentity(sequential_7/dense_20/Relu:activations:0*
T0*'
_output_shapes
:         а
+sequential_7/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_21_matmul_readvariableop_resource*
_output_shapes

:*
dtype0И
sequential_7/dense_21/MatMulMatMul)sequential_7/dropout_37/Identity:output:03sequential_7/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ъ
,sequential_7/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0И
sequential_7/dense_21/BiasAddBiasAdd&sequential_7/dense_21/MatMul:product:04sequential_7/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         |
sequential_7/dense_21/ReluRelu&sequential_7/dense_21/BiasAdd:output:0*
T0*'
_output_shapes
:         ѕ
 sequential_7/dropout_38/IdentityIdentity(sequential_7/dense_21/Relu:activations:0*
T0*'
_output_shapes
:         а
+sequential_7/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_22_matmul_readvariableop_resource*
_output_shapes

:*
dtype0И
sequential_7/dense_22/MatMulMatMul)sequential_7/dropout_38/Identity:output:03sequential_7/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ъ
,sequential_7/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0И
sequential_7/dense_22/BiasAddBiasAdd&sequential_7/dense_22/MatMul:product:04sequential_7/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         u
IdentityIdentity&sequential_7/dense_22/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ц
NoOpNoOp-^sequential_7/dense_19/BiasAdd/ReadVariableOp,^sequential_7/dense_19/MatMul/ReadVariableOp-^sequential_7/dense_20/BiasAdd/ReadVariableOp,^sequential_7/dense_20/MatMul/ReadVariableOp-^sequential_7/dense_21/BiasAdd/ReadVariableOp,^sequential_7/dense_21/MatMul/ReadVariableOp-^sequential_7/dense_22/BiasAdd/ReadVariableOp,^sequential_7/dense_22/MatMul/ReadVariableOp9^sequential_7/lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp8^sequential_7/lstm_21/lstm_cell_21/MatMul/ReadVariableOp:^sequential_7/lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp^sequential_7/lstm_21/while9^sequential_7/lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp8^sequential_7/lstm_22/lstm_cell_22/MatMul/ReadVariableOp:^sequential_7/lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp^sequential_7/lstm_22/while9^sequential_7/lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp8^sequential_7/lstm_23/lstm_cell_23/MatMul/ReadVariableOp:^sequential_7/lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp^sequential_7/lstm_23/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         : : : : : : : : : : : : : : : : : 2\
,sequential_7/dense_19/BiasAdd/ReadVariableOp,sequential_7/dense_19/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_19/MatMul/ReadVariableOp+sequential_7/dense_19/MatMul/ReadVariableOp2\
,sequential_7/dense_20/BiasAdd/ReadVariableOp,sequential_7/dense_20/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_20/MatMul/ReadVariableOp+sequential_7/dense_20/MatMul/ReadVariableOp2\
,sequential_7/dense_21/BiasAdd/ReadVariableOp,sequential_7/dense_21/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_21/MatMul/ReadVariableOp+sequential_7/dense_21/MatMul/ReadVariableOp2\
,sequential_7/dense_22/BiasAdd/ReadVariableOp,sequential_7/dense_22/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_22/MatMul/ReadVariableOp+sequential_7/dense_22/MatMul/ReadVariableOp2t
8sequential_7/lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp8sequential_7/lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp2r
7sequential_7/lstm_21/lstm_cell_21/MatMul/ReadVariableOp7sequential_7/lstm_21/lstm_cell_21/MatMul/ReadVariableOp2v
9sequential_7/lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp9sequential_7/lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp28
sequential_7/lstm_21/whilesequential_7/lstm_21/while2t
8sequential_7/lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp8sequential_7/lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp2r
7sequential_7/lstm_22/lstm_cell_22/MatMul/ReadVariableOp7sequential_7/lstm_22/lstm_cell_22/MatMul/ReadVariableOp2v
9sequential_7/lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp9sequential_7/lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp28
sequential_7/lstm_22/whilesequential_7/lstm_22/while2t
8sequential_7/lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp8sequential_7/lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp2r
7sequential_7/lstm_23/lstm_cell_23/MatMul/ReadVariableOp7sequential_7/lstm_23/lstm_cell_23/MatMul/ReadVariableOp2v
9sequential_7/lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp9sequential_7/lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp28
sequential_7/lstm_23/whilesequential_7/lstm_23/while:Z V
+
_output_shapes
:         
'
_user_specified_namelstm_21_input
 7
╩
while_body_681959
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_22_matmul_readvariableop_resource_0:@G
5while_lstm_cell_22_matmul_1_readvariableop_resource_0:@B
4while_lstm_cell_22_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_22_matmul_readvariableop_resource:@E
3while_lstm_cell_22_matmul_1_readvariableop_resource:@@
2while_lstm_cell_22_biasadd_readvariableop_resource:@ѕб)while/lstm_cell_22/BiasAdd/ReadVariableOpб(while/lstm_cell_22/MatMul/ReadVariableOpб*while/lstm_cell_22/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ю
(while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_22_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0╣
while/lstm_cell_22/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @а
*while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_22_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0а
while/lstm_cell_22/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_22/addAddV2#while/lstm_cell_22/MatMul:product:0%while/lstm_cell_22/MatMul_1:product:0*
T0*'
_output_shapes
:         @џ
)while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0д
while/lstm_cell_22/BiasAddBiasAddwhile/lstm_cell_22/add:z:01while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0#while/lstm_cell_22/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitz
while/lstm_cell_22/SigmoidSigmoid!while/lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         |
while/lstm_cell_22/Sigmoid_1Sigmoid!while/lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         є
while/lstm_cell_22/mulMul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         t
while/lstm_cell_22/ReluRelu!while/lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         ў
while/lstm_cell_22/mul_1Mulwhile/lstm_cell_22/Sigmoid:y:0%while/lstm_cell_22/Relu:activations:0*
T0*'
_output_shapes
:         Ї
while/lstm_cell_22/add_1AddV2while/lstm_cell_22/mul:z:0while/lstm_cell_22/mul_1:z:0*
T0*'
_output_shapes
:         |
while/lstm_cell_22/Sigmoid_2Sigmoid!while/lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         q
while/lstm_cell_22/Relu_1Reluwhile/lstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         ю
while/lstm_cell_22/mul_2Mul while/lstm_cell_22/Sigmoid_2:y:0'while/lstm_cell_22/Relu_1:activations:0*
T0*'
_output_shapes
:         ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_22/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_22/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         y
while/Identity_5Identitywhile/lstm_cell_22/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         л

while/NoOpNoOp*^while/lstm_cell_22/BiasAdd/ReadVariableOp)^while/lstm_cell_22/MatMul/ReadVariableOp+^while/lstm_cell_22/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_22_biasadd_readvariableop_resource4while_lstm_cell_22_biasadd_readvariableop_resource_0"l
3while_lstm_cell_22_matmul_1_readvariableop_resource5while_lstm_cell_22_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_22_matmul_readvariableop_resource3while_lstm_cell_22_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2V
)while/lstm_cell_22/BiasAdd/ReadVariableOp)while/lstm_cell_22/BiasAdd/ReadVariableOp2T
(while/lstm_cell_22/MatMul/ReadVariableOp(while/lstm_cell_22/MatMul/ReadVariableOp2X
*while/lstm_cell_22/MatMul_1/ReadVariableOp*while/lstm_cell_22/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
ИA
╩

lstm_22_while_body_683295,
(lstm_22_while_lstm_22_while_loop_counter2
.lstm_22_while_lstm_22_while_maximum_iterations
lstm_22_while_placeholder
lstm_22_while_placeholder_1
lstm_22_while_placeholder_2
lstm_22_while_placeholder_3+
'lstm_22_while_lstm_22_strided_slice_1_0g
clstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0M
;lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0:@O
=lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0:@J
<lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0:@
lstm_22_while_identity
lstm_22_while_identity_1
lstm_22_while_identity_2
lstm_22_while_identity_3
lstm_22_while_identity_4
lstm_22_while_identity_5)
%lstm_22_while_lstm_22_strided_slice_1e
alstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensorK
9lstm_22_while_lstm_cell_22_matmul_readvariableop_resource:@M
;lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource:@H
:lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource:@ѕб1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOpб0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOpб2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpљ
?lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╬
1lstm_22/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0lstm_22_while_placeholderHlstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0г
0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp;lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0Л
!lstm_22/while/lstm_cell_22/MatMulMatMul8lstm_22/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @░
2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp=lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0И
#lstm_22/while/lstm_cell_22/MatMul_1MatMullstm_22_while_placeholder_2:lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @х
lstm_22/while/lstm_cell_22/addAddV2+lstm_22/while/lstm_cell_22/MatMul:product:0-lstm_22/while/lstm_cell_22/MatMul_1:product:0*
T0*'
_output_shapes
:         @ф
1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp<lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Й
"lstm_22/while/lstm_cell_22/BiasAddBiasAdd"lstm_22/while/lstm_cell_22/add:z:09lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @l
*lstm_22/while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Є
 lstm_22/while/lstm_cell_22/splitSplit3lstm_22/while/lstm_cell_22/split/split_dim:output:0+lstm_22/while/lstm_cell_22/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitі
"lstm_22/while/lstm_cell_22/SigmoidSigmoid)lstm_22/while/lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         ї
$lstm_22/while/lstm_cell_22/Sigmoid_1Sigmoid)lstm_22/while/lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         ъ
lstm_22/while/lstm_cell_22/mulMul(lstm_22/while/lstm_cell_22/Sigmoid_1:y:0lstm_22_while_placeholder_3*
T0*'
_output_shapes
:         ё
lstm_22/while/lstm_cell_22/ReluRelu)lstm_22/while/lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         ░
 lstm_22/while/lstm_cell_22/mul_1Mul&lstm_22/while/lstm_cell_22/Sigmoid:y:0-lstm_22/while/lstm_cell_22/Relu:activations:0*
T0*'
_output_shapes
:         Ц
 lstm_22/while/lstm_cell_22/add_1AddV2"lstm_22/while/lstm_cell_22/mul:z:0$lstm_22/while/lstm_cell_22/mul_1:z:0*
T0*'
_output_shapes
:         ї
$lstm_22/while/lstm_cell_22/Sigmoid_2Sigmoid)lstm_22/while/lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         Ђ
!lstm_22/while/lstm_cell_22/Relu_1Relu$lstm_22/while/lstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         ┤
 lstm_22/while/lstm_cell_22/mul_2Mul(lstm_22/while/lstm_cell_22/Sigmoid_2:y:0/lstm_22/while/lstm_cell_22/Relu_1:activations:0*
T0*'
_output_shapes
:         т
2lstm_22/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_22_while_placeholder_1lstm_22_while_placeholder$lstm_22/while/lstm_cell_22/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмU
lstm_22/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_22/while/addAddV2lstm_22_while_placeholderlstm_22/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_22/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Є
lstm_22/while/add_1AddV2(lstm_22_while_lstm_22_while_loop_counterlstm_22/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_22/while/IdentityIdentitylstm_22/while/add_1:z:0^lstm_22/while/NoOp*
T0*
_output_shapes
: і
lstm_22/while/Identity_1Identity.lstm_22_while_lstm_22_while_maximum_iterations^lstm_22/while/NoOp*
T0*
_output_shapes
: q
lstm_22/while/Identity_2Identitylstm_22/while/add:z:0^lstm_22/while/NoOp*
T0*
_output_shapes
: ъ
lstm_22/while/Identity_3IdentityBlstm_22/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_22/while/NoOp*
T0*
_output_shapes
: Љ
lstm_22/while/Identity_4Identity$lstm_22/while/lstm_cell_22/mul_2:z:0^lstm_22/while/NoOp*
T0*'
_output_shapes
:         Љ
lstm_22/while/Identity_5Identity$lstm_22/while/lstm_cell_22/add_1:z:0^lstm_22/while/NoOp*
T0*'
_output_shapes
:         ­
lstm_22/while/NoOpNoOp2^lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp1^lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp3^lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_22_while_identitylstm_22/while/Identity:output:0"=
lstm_22_while_identity_1!lstm_22/while/Identity_1:output:0"=
lstm_22_while_identity_2!lstm_22/while/Identity_2:output:0"=
lstm_22_while_identity_3!lstm_22/while/Identity_3:output:0"=
lstm_22_while_identity_4!lstm_22/while/Identity_4:output:0"=
lstm_22_while_identity_5!lstm_22/while/Identity_5:output:0"P
%lstm_22_while_lstm_22_strided_slice_1'lstm_22_while_lstm_22_strided_slice_1_0"z
:lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource<lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0"|
;lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource=lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0"x
9lstm_22_while_lstm_cell_22_matmul_readvariableop_resource;lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0"╚
alstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensorclstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2f
1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp2d
0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp2h
2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
э
▓
(__inference_lstm_23_layer_call_fn_684904

inputs
unknown:@
	unknown_0:@
	unknown_1:@
identityѕбStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_lstm_23_layer_call_and_return_conditional_losses_681426o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ќx
┌
__inference__traced_save_686171
file_prefix.
*savev2_dense_19_kernel_read_readvariableop,
(savev2_dense_19_bias_read_readvariableop.
*savev2_dense_20_kernel_read_readvariableop,
(savev2_dense_20_bias_read_readvariableop.
*savev2_dense_21_kernel_read_readvariableop,
(savev2_dense_21_bias_read_readvariableop.
*savev2_dense_22_kernel_read_readvariableop,
(savev2_dense_22_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_21_lstm_cell_21_kernel_read_readvariableopD
@savev2_lstm_21_lstm_cell_21_recurrent_kernel_read_readvariableop8
4savev2_lstm_21_lstm_cell_21_bias_read_readvariableop:
6savev2_lstm_22_lstm_cell_22_kernel_read_readvariableopD
@savev2_lstm_22_lstm_cell_22_recurrent_kernel_read_readvariableop8
4savev2_lstm_22_lstm_cell_22_bias_read_readvariableop:
6savev2_lstm_23_lstm_cell_23_kernel_read_readvariableopD
@savev2_lstm_23_lstm_cell_23_recurrent_kernel_read_readvariableop8
4savev2_lstm_23_lstm_cell_23_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_19_kernel_m_read_readvariableop3
/savev2_adam_dense_19_bias_m_read_readvariableop5
1savev2_adam_dense_20_kernel_m_read_readvariableop3
/savev2_adam_dense_20_bias_m_read_readvariableop5
1savev2_adam_dense_21_kernel_m_read_readvariableop3
/savev2_adam_dense_21_bias_m_read_readvariableop5
1savev2_adam_dense_22_kernel_m_read_readvariableop3
/savev2_adam_dense_22_bias_m_read_readvariableopA
=savev2_adam_lstm_21_lstm_cell_21_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_21_lstm_cell_21_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_21_lstm_cell_21_bias_m_read_readvariableopA
=savev2_adam_lstm_22_lstm_cell_22_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_22_lstm_cell_22_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_22_lstm_cell_22_bias_m_read_readvariableopA
=savev2_adam_lstm_23_lstm_cell_23_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_23_lstm_cell_23_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_23_lstm_cell_23_bias_m_read_readvariableop5
1savev2_adam_dense_19_kernel_v_read_readvariableop3
/savev2_adam_dense_19_bias_v_read_readvariableop5
1savev2_adam_dense_20_kernel_v_read_readvariableop3
/savev2_adam_dense_20_bias_v_read_readvariableop5
1savev2_adam_dense_21_kernel_v_read_readvariableop3
/savev2_adam_dense_21_bias_v_read_readvariableop5
1savev2_adam_dense_22_kernel_v_read_readvariableop3
/savev2_adam_dense_22_bias_v_read_readvariableopA
=savev2_adam_lstm_21_lstm_cell_21_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_21_lstm_cell_21_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_21_lstm_cell_21_bias_v_read_readvariableopA
=savev2_adam_lstm_22_lstm_cell_22_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_22_lstm_cell_22_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_22_lstm_cell_22_bias_v_read_readvariableopA
=savev2_adam_lstm_23_lstm_cell_23_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_23_lstm_cell_23_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_23_lstm_cell_23_bias_v_read_readvariableop
savev2_const

identity_1ѕбMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partЂ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ▀
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*ѕ
value■Bч=B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЖ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*Ј
valueЁBѓ=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ж
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_19_kernel_read_readvariableop(savev2_dense_19_bias_read_readvariableop*savev2_dense_20_kernel_read_readvariableop(savev2_dense_20_bias_read_readvariableop*savev2_dense_21_kernel_read_readvariableop(savev2_dense_21_bias_read_readvariableop*savev2_dense_22_kernel_read_readvariableop(savev2_dense_22_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_21_lstm_cell_21_kernel_read_readvariableop@savev2_lstm_21_lstm_cell_21_recurrent_kernel_read_readvariableop4savev2_lstm_21_lstm_cell_21_bias_read_readvariableop6savev2_lstm_22_lstm_cell_22_kernel_read_readvariableop@savev2_lstm_22_lstm_cell_22_recurrent_kernel_read_readvariableop4savev2_lstm_22_lstm_cell_22_bias_read_readvariableop6savev2_lstm_23_lstm_cell_23_kernel_read_readvariableop@savev2_lstm_23_lstm_cell_23_recurrent_kernel_read_readvariableop4savev2_lstm_23_lstm_cell_23_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_19_kernel_m_read_readvariableop/savev2_adam_dense_19_bias_m_read_readvariableop1savev2_adam_dense_20_kernel_m_read_readvariableop/savev2_adam_dense_20_bias_m_read_readvariableop1savev2_adam_dense_21_kernel_m_read_readvariableop/savev2_adam_dense_21_bias_m_read_readvariableop1savev2_adam_dense_22_kernel_m_read_readvariableop/savev2_adam_dense_22_bias_m_read_readvariableop=savev2_adam_lstm_21_lstm_cell_21_kernel_m_read_readvariableopGsavev2_adam_lstm_21_lstm_cell_21_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_21_lstm_cell_21_bias_m_read_readvariableop=savev2_adam_lstm_22_lstm_cell_22_kernel_m_read_readvariableopGsavev2_adam_lstm_22_lstm_cell_22_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_22_lstm_cell_22_bias_m_read_readvariableop=savev2_adam_lstm_23_lstm_cell_23_kernel_m_read_readvariableopGsavev2_adam_lstm_23_lstm_cell_23_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_23_lstm_cell_23_bias_m_read_readvariableop1savev2_adam_dense_19_kernel_v_read_readvariableop/savev2_adam_dense_19_bias_v_read_readvariableop1savev2_adam_dense_20_kernel_v_read_readvariableop/savev2_adam_dense_20_bias_v_read_readvariableop1savev2_adam_dense_21_kernel_v_read_readvariableop/savev2_adam_dense_21_bias_v_read_readvariableop1savev2_adam_dense_22_kernel_v_read_readvariableop/savev2_adam_dense_22_bias_v_read_readvariableop=savev2_adam_lstm_21_lstm_cell_21_kernel_v_read_readvariableopGsavev2_adam_lstm_21_lstm_cell_21_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_21_lstm_cell_21_bias_v_read_readvariableop=savev2_adam_lstm_22_lstm_cell_22_kernel_v_read_readvariableopGsavev2_adam_lstm_22_lstm_cell_22_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_22_lstm_cell_22_bias_v_read_readvariableop=savev2_adam_lstm_23_lstm_cell_23_kernel_v_read_readvariableopGsavev2_adam_lstm_23_lstm_cell_23_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_23_lstm_cell_23_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *K
dtypesA
?2=	љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:І
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Н
_input_shapes├
└: ::::::::: : : : : :@:@:@:@:@:@:@:@:@: : : : :::::::::@:@:@:@:@:@:@:@:@:::::::::@:@:@:@:@:@:@:@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@:$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@:$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@:$ 

_output_shapes

:@: 

_output_shapes
:@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

::  

_output_shapes
::$! 

_output_shapes

:: "

_output_shapes
::$# 

_output_shapes

:@:$$ 

_output_shapes

:@: %

_output_shapes
:@:$& 

_output_shapes

:@:$' 

_output_shapes

:@: (

_output_shapes
:@:$) 

_output_shapes

:@:$* 

_output_shapes

:@: +

_output_shapes
:@:$, 

_output_shapes

:: -

_output_shapes
::$. 

_output_shapes

:: /

_output_shapes
::$0 

_output_shapes

:: 1

_output_shapes
::$2 

_output_shapes

:: 3

_output_shapes
::$4 

_output_shapes

:@:$5 

_output_shapes

:@: 6

_output_shapes
:@:$7 

_output_shapes

:@:$8 

_output_shapes

:@: 9

_output_shapes
:@:$: 

_output_shapes

:@:$; 

_output_shapes

:@: <

_output_shapes
:@:=

_output_shapes
: 
К	
ш
D__inference_dense_22_layer_call_and_return_conditional_losses_681523

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
х
├
while_cond_680184
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_680184___redundant_placeholder04
0while_while_cond_680184___redundant_placeholder14
0while_while_cond_680184___redundant_placeholder24
0while_while_cond_680184___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
ж
d
F__inference_dropout_33_layer_call_and_return_conditional_losses_684216

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         _

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         :S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Ы
┬
-__inference_sequential_7_layer_call_fn_681567
lstm_21_input
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:
identityѕбStatefulPartitionedCall░
StatefulPartitionedCallStatefulPartitionedCalllstm_21_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *3
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_681530o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:         
'
_user_specified_namelstm_21_input
З	
e
F__inference_dropout_36_layer_call_and_return_conditional_losses_685561

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
х
├
while_cond_682146
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_682146___redundant_placeholder04
0while_while_cond_682146___redundant_placeholder14
0while_while_cond_682146___redundant_placeholder24
0while_while_cond_682146___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
А7
ј
H__inference_sequential_7_layer_call_and_return_conditional_losses_682459
lstm_21_input 
lstm_21_682411:@ 
lstm_21_682413:@
lstm_21_682415:@ 
lstm_22_682419:@ 
lstm_22_682421:@
lstm_22_682423:@ 
lstm_23_682427:@ 
lstm_23_682429:@
lstm_23_682431:@!
dense_19_682435:
dense_19_682437:!
dense_20_682441:
dense_20_682443:!
dense_21_682447:
dense_21_682449:!
dense_22_682453:
dense_22_682455:
identityѕб dense_19/StatefulPartitionedCallб dense_20/StatefulPartitionedCallб dense_21/StatefulPartitionedCallб dense_22/StatefulPartitionedCallбlstm_21/StatefulPartitionedCallбlstm_22/StatefulPartitionedCallбlstm_23/StatefulPartitionedCallї
lstm_21/StatefulPartitionedCallStatefulPartitionedCalllstm_21_inputlstm_21_682411lstm_21_682413lstm_21_682415*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_lstm_21_layer_call_and_return_conditional_losses_681112т
dropout_33/PartitionedCallPartitionedCall(lstm_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_33_layer_call_and_return_conditional_losses_681125б
lstm_22/StatefulPartitionedCallStatefulPartitionedCall#dropout_33/PartitionedCall:output:0lstm_22_682419lstm_22_682421lstm_22_682423*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_681269т
dropout_34/PartitionedCallPartitionedCall(lstm_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_34_layer_call_and_return_conditional_losses_681282ъ
lstm_23/StatefulPartitionedCallStatefulPartitionedCall#dropout_34/PartitionedCall:output:0lstm_23_682427lstm_23_682429lstm_23_682431*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_lstm_23_layer_call_and_return_conditional_losses_681426р
dropout_35/PartitionedCallPartitionedCall(lstm_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_35_layer_call_and_return_conditional_losses_681439љ
 dense_19/StatefulPartitionedCallStatefulPartitionedCall#dropout_35/PartitionedCall:output:0dense_19_682435dense_19_682437*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_681452Р
dropout_36/PartitionedCallPartitionedCall)dense_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_681463љ
 dense_20/StatefulPartitionedCallStatefulPartitionedCall#dropout_36/PartitionedCall:output:0dense_20_682441dense_20_682443*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_681476Р
dropout_37/PartitionedCallPartitionedCall)dense_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_681487љ
 dense_21/StatefulPartitionedCallStatefulPartitionedCall#dropout_37/PartitionedCall:output:0dense_21_682447dense_21_682449*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_681500Р
dropout_38/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_681511љ
 dense_22/StatefulPartitionedCallStatefulPartitionedCall#dropout_38/PartitionedCall:output:0dense_22_682453dense_22_682455*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_681523x
IdentityIdentity)dense_22/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         И
NoOpNoOp!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall ^lstm_21/StatefulPartitionedCall ^lstm_22/StatefulPartitionedCall ^lstm_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         : : : : : : : : : : : : : : : : : 2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2B
lstm_21/StatefulPartitionedCalllstm_21/StatefulPartitionedCall2B
lstm_22/StatefulPartitionedCalllstm_22/StatefulPartitionedCall2B
lstm_23/StatefulPartitionedCalllstm_23/StatefulPartitionedCall:Z V
+
_output_shapes
:         
'
_user_specified_namelstm_21_input
Ј
┤
(__inference_lstm_23_layer_call_fn_684893
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
identityѕбStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_lstm_23_layer_call_and_return_conditional_losses_680954o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
х
├
while_cond_685402
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_685402___redundant_placeholder04
0while_while_cond_685402___redundant_placeholder14
0while_while_cond_685402___redundant_placeholder24
0while_while_cond_685402___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
┘
d
F__inference_dropout_36_layer_call_and_return_conditional_losses_681463

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╦
Ђ
H__inference_lstm_cell_21_layer_call_and_return_conditional_losses_679980

inputs

states
states_10
matmul_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         Љ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_namestates:OK
'
_output_shapes
:         
 
_user_specified_namestates
┬
╣
$__inference_signature_wrapper_682557
lstm_21_input
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:
identityѕбStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCalllstm_21_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *3
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ **
f%R#
!__inference__wrapped_model_679913o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:         
'
_user_specified_namelstm_21_input
М
Ѓ
H__inference_lstm_cell_23_layer_call_and_return_conditional_losses_685968

inputs
states_0
states_10
matmul_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         Љ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
states/0:QM
'
_output_shapes
:         
"
_user_specified_name
states/1
ж
d
F__inference_dropout_34_layer_call_and_return_conditional_losses_681282

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         _

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         :S O
+
_output_shapes
:         
 
_user_specified_nameinputs
х
├
while_cond_684759
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_684759___redundant_placeholder04
0while_while_cond_684759___redundant_placeholder14
0while_while_cond_684759___redundant_placeholder24
0while_while_cond_684759___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
Џ

ш
D__inference_dense_19_layer_call_and_return_conditional_losses_681452

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┘
d
F__inference_dropout_35_layer_call_and_return_conditional_losses_685502

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
 
▓
(__inference_lstm_22_layer_call_fn_684261

inputs
unknown:@
	unknown_0:@
	unknown_1:@
identityѕбStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_681269s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
х
├
while_cond_680534
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_680534___redundant_placeholder04
0while_while_cond_680534___redundant_placeholder14
0while_while_cond_680534___redundant_placeholder24
0while_while_cond_680534___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
 7
╩
while_body_683688
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_21_matmul_readvariableop_resource_0:@G
5while_lstm_cell_21_matmul_1_readvariableop_resource_0:@B
4while_lstm_cell_21_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_21_matmul_readvariableop_resource:@E
3while_lstm_cell_21_matmul_1_readvariableop_resource:@@
2while_lstm_cell_21_biasadd_readvariableop_resource:@ѕб)while/lstm_cell_21/BiasAdd/ReadVariableOpб(while/lstm_cell_21/MatMul/ReadVariableOpб*while/lstm_cell_21/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ю
(while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_21_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0╣
while/lstm_cell_21/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @а
*while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_21_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0а
while/lstm_cell_21/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_21/addAddV2#while/lstm_cell_21/MatMul:product:0%while/lstm_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:         @џ
)while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0д
while/lstm_cell_21/BiasAddBiasAddwhile/lstm_cell_21/add:z:01while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
"while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_21/splitSplit+while/lstm_cell_21/split/split_dim:output:0#while/lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitz
while/lstm_cell_21/SigmoidSigmoid!while/lstm_cell_21/split:output:0*
T0*'
_output_shapes
:         |
while/lstm_cell_21/Sigmoid_1Sigmoid!while/lstm_cell_21/split:output:1*
T0*'
_output_shapes
:         є
while/lstm_cell_21/mulMul while/lstm_cell_21/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         t
while/lstm_cell_21/ReluRelu!while/lstm_cell_21/split:output:2*
T0*'
_output_shapes
:         ў
while/lstm_cell_21/mul_1Mulwhile/lstm_cell_21/Sigmoid:y:0%while/lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         Ї
while/lstm_cell_21/add_1AddV2while/lstm_cell_21/mul:z:0while/lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         |
while/lstm_cell_21/Sigmoid_2Sigmoid!while/lstm_cell_21/split:output:3*
T0*'
_output_shapes
:         q
while/lstm_cell_21/Relu_1Reluwhile/lstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         ю
while/lstm_cell_21/mul_2Mul while/lstm_cell_21/Sigmoid_2:y:0'while/lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_21/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_21/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         y
while/Identity_5Identitywhile/lstm_cell_21/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         л

while/NoOpNoOp*^while/lstm_cell_21/BiasAdd/ReadVariableOp)^while/lstm_cell_21/MatMul/ReadVariableOp+^while/lstm_cell_21/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_21_biasadd_readvariableop_resource4while_lstm_cell_21_biasadd_readvariableop_resource_0"l
3while_lstm_cell_21_matmul_1_readvariableop_resource5while_lstm_cell_21_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_21_matmul_readvariableop_resource3while_lstm_cell_21_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2V
)while/lstm_cell_21/BiasAdd/ReadVariableOp)while/lstm_cell_21/BiasAdd/ReadVariableOp2T
(while/lstm_cell_21/MatMul/ReadVariableOp(while/lstm_cell_21/MatMul/ReadVariableOp2X
*while/lstm_cell_21/MatMul_1/ReadVariableOp*while/lstm_cell_21/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
х
├
while_cond_681341
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_681341___redundant_placeholder04
0while_while_cond_681341___redundant_placeholder14
0while_while_cond_681341___redundant_placeholder24
0while_while_cond_681341___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
Џ

ш
D__inference_dense_21_layer_call_and_return_conditional_losses_681500

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┘
d
F__inference_dropout_37_layer_call_and_return_conditional_losses_685596

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
З	
e
F__inference_dropout_36_layer_call_and_return_conditional_losses_681663

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
х
├
while_cond_684116
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_684116___redundant_placeholder04
0while_while_cond_684116___redundant_placeholder14
0while_while_cond_684116___redundant_placeholder24
0while_while_cond_684116___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
ѓ
у
&sequential_7_lstm_22_while_cond_679657F
Bsequential_7_lstm_22_while_sequential_7_lstm_22_while_loop_counterL
Hsequential_7_lstm_22_while_sequential_7_lstm_22_while_maximum_iterations*
&sequential_7_lstm_22_while_placeholder,
(sequential_7_lstm_22_while_placeholder_1,
(sequential_7_lstm_22_while_placeholder_2,
(sequential_7_lstm_22_while_placeholder_3H
Dsequential_7_lstm_22_while_less_sequential_7_lstm_22_strided_slice_1^
Zsequential_7_lstm_22_while_sequential_7_lstm_22_while_cond_679657___redundant_placeholder0^
Zsequential_7_lstm_22_while_sequential_7_lstm_22_while_cond_679657___redundant_placeholder1^
Zsequential_7_lstm_22_while_sequential_7_lstm_22_while_cond_679657___redundant_placeholder2^
Zsequential_7_lstm_22_while_sequential_7_lstm_22_while_cond_679657___redundant_placeholder3'
#sequential_7_lstm_22_while_identity
Х
sequential_7/lstm_22/while/LessLess&sequential_7_lstm_22_while_placeholderDsequential_7_lstm_22_while_less_sequential_7_lstm_22_strided_slice_1*
T0*
_output_shapes
: u
#sequential_7/lstm_22/while/IdentityIdentity#sequential_7/lstm_22/while/Less:z:0*
T0
*
_output_shapes
: "S
#sequential_7_lstm_22_while_identity,sequential_7/lstm_22/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
њ8
ѓ
C__inference_lstm_21_layer_call_and_return_conditional_losses_680063

inputs%
lstm_cell_21_679981:@%
lstm_cell_21_679983:@!
lstm_cell_21_679985:@
identityѕб$lstm_cell_21/StatefulPartitionedCallбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЭ
$lstm_cell_21/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_21_679981lstm_cell_21_679983lstm_cell_21_679985*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_lstm_cell_21_layer_call_and_return_conditional_losses_679980n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : и
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_21_679981lstm_cell_21_679983lstm_cell_21_679985*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_679994*
condR
while_cond_679993*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  u
NoOpNoOp%^lstm_cell_21/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2L
$lstm_cell_21/StatefulPartitionedCall$lstm_cell_21/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
 7
╩
while_body_683831
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_21_matmul_readvariableop_resource_0:@G
5while_lstm_cell_21_matmul_1_readvariableop_resource_0:@B
4while_lstm_cell_21_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_21_matmul_readvariableop_resource:@E
3while_lstm_cell_21_matmul_1_readvariableop_resource:@@
2while_lstm_cell_21_biasadd_readvariableop_resource:@ѕб)while/lstm_cell_21/BiasAdd/ReadVariableOpб(while/lstm_cell_21/MatMul/ReadVariableOpб*while/lstm_cell_21/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ю
(while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_21_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0╣
while/lstm_cell_21/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @а
*while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_21_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0а
while/lstm_cell_21/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_21/addAddV2#while/lstm_cell_21/MatMul:product:0%while/lstm_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:         @џ
)while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0д
while/lstm_cell_21/BiasAddBiasAddwhile/lstm_cell_21/add:z:01while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
"while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_21/splitSplit+while/lstm_cell_21/split/split_dim:output:0#while/lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitz
while/lstm_cell_21/SigmoidSigmoid!while/lstm_cell_21/split:output:0*
T0*'
_output_shapes
:         |
while/lstm_cell_21/Sigmoid_1Sigmoid!while/lstm_cell_21/split:output:1*
T0*'
_output_shapes
:         є
while/lstm_cell_21/mulMul while/lstm_cell_21/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         t
while/lstm_cell_21/ReluRelu!while/lstm_cell_21/split:output:2*
T0*'
_output_shapes
:         ў
while/lstm_cell_21/mul_1Mulwhile/lstm_cell_21/Sigmoid:y:0%while/lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         Ї
while/lstm_cell_21/add_1AddV2while/lstm_cell_21/mul:z:0while/lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         |
while/lstm_cell_21/Sigmoid_2Sigmoid!while/lstm_cell_21/split:output:3*
T0*'
_output_shapes
:         q
while/lstm_cell_21/Relu_1Reluwhile/lstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         ю
while/lstm_cell_21/mul_2Mul while/lstm_cell_21/Sigmoid_2:y:0'while/lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_21/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_21/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         y
while/Identity_5Identitywhile/lstm_cell_21/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         л

while/NoOpNoOp*^while/lstm_cell_21/BiasAdd/ReadVariableOp)^while/lstm_cell_21/MatMul/ReadVariableOp+^while/lstm_cell_21/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_21_biasadd_readvariableop_resource4while_lstm_cell_21_biasadd_readvariableop_resource_0"l
3while_lstm_cell_21_matmul_1_readvariableop_resource5while_lstm_cell_21_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_21_matmul_readvariableop_resource3while_lstm_cell_21_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2V
)while/lstm_cell_21/BiasAdd/ReadVariableOp)while/lstm_cell_21/BiasAdd/ReadVariableOp2T
(while/lstm_cell_21/MatMul/ReadVariableOp(while/lstm_cell_21/MatMul/ReadVariableOp2X
*while/lstm_cell_21/MatMul_1/ReadVariableOp*while/lstm_cell_21/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
МJ
џ
C__inference_lstm_23_layer_call_and_return_conditional_losses_685201
inputs_0=
+lstm_cell_23_matmul_readvariableop_resource:@?
-lstm_cell_23_matmul_1_readvariableop_resource:@:
,lstm_cell_23_biasadd_readvariableop_resource:@
identityѕб#lstm_cell_23/BiasAdd/ReadVariableOpб"lstm_cell_23/MatMul/ReadVariableOpб$lstm_cell_23/MatMul_1/ReadVariableOpбwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskј
"lstm_cell_23/MatMul/ReadVariableOpReadVariableOp+lstm_cell_23_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ћ
lstm_cell_23/MatMulMatMulstrided_slice_2:output:0*lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @њ
$lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_23_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell_23/MatMul_1MatMulzeros:output:0,lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @І
lstm_cell_23/addAddV2lstm_cell_23/MatMul:product:0lstm_cell_23/MatMul_1:product:0*
T0*'
_output_shapes
:         @ї
#lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ћ
lstm_cell_23/BiasAddBiasAddlstm_cell_23/add:z:0+lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @^
lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_23/splitSplit%lstm_cell_23/split/split_dim:output:0lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitn
lstm_cell_23/SigmoidSigmoidlstm_cell_23/split:output:0*
T0*'
_output_shapes
:         p
lstm_cell_23/Sigmoid_1Sigmoidlstm_cell_23/split:output:1*
T0*'
_output_shapes
:         w
lstm_cell_23/mulMullstm_cell_23/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         h
lstm_cell_23/ReluRelulstm_cell_23/split:output:2*
T0*'
_output_shapes
:         є
lstm_cell_23/mul_1Mullstm_cell_23/Sigmoid:y:0lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         {
lstm_cell_23/add_1AddV2lstm_cell_23/mul:z:0lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         p
lstm_cell_23/Sigmoid_2Sigmoidlstm_cell_23/split:output:3*
T0*'
_output_shapes
:         e
lstm_cell_23/Relu_1Relulstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         і
lstm_cell_23/mul_2Mullstm_cell_23/Sigmoid_2:y:0!lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѓ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_23_matmul_readvariableop_resource-lstm_cell_23_matmul_1_readvariableop_resource,lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_685117*
condR
while_cond_685116*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         └
NoOpNoOp$^lstm_cell_23/BiasAdd/ReadVariableOp#^lstm_cell_23/MatMul/ReadVariableOp%^lstm_cell_23/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2J
#lstm_cell_23/BiasAdd/ReadVariableOp#lstm_cell_23/BiasAdd/ReadVariableOp2H
"lstm_cell_23/MatMul/ReadVariableOp"lstm_cell_23/MatMul/ReadVariableOp2L
$lstm_cell_23/MatMul_1/ReadVariableOp$lstm_cell_23/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
њ8
ѓ
C__inference_lstm_22_layer_call_and_return_conditional_losses_680413

inputs%
lstm_cell_22_680331:@%
lstm_cell_22_680333:@!
lstm_cell_22_680335:@
identityѕб$lstm_cell_22/StatefulPartitionedCallбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЭ
$lstm_cell_22/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_22_680331lstm_cell_22_680333lstm_cell_22_680335*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_680330n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : и
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_22_680331lstm_cell_22_680333lstm_cell_22_680335*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_680344*
condR
while_cond_680343*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  u
NoOpNoOp%^lstm_cell_22/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2L
$lstm_cell_22/StatefulPartitionedCall$lstm_cell_22/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
ц
G
+__inference_dropout_37_layer_call_fn_685586

inputs
identity┤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_681487`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ќ

с
lstm_23_while_cond_683441,
(lstm_23_while_lstm_23_while_loop_counter2
.lstm_23_while_lstm_23_while_maximum_iterations
lstm_23_while_placeholder
lstm_23_while_placeholder_1
lstm_23_while_placeholder_2
lstm_23_while_placeholder_3.
*lstm_23_while_less_lstm_23_strided_slice_1D
@lstm_23_while_lstm_23_while_cond_683441___redundant_placeholder0D
@lstm_23_while_lstm_23_while_cond_683441___redundant_placeholder1D
@lstm_23_while_lstm_23_while_cond_683441___redundant_placeholder2D
@lstm_23_while_lstm_23_while_cond_683441___redundant_placeholder3
lstm_23_while_identity
ѓ
lstm_23/while/LessLesslstm_23_while_placeholder*lstm_23_while_less_lstm_23_strided_slice_1*
T0*
_output_shapes
: [
lstm_23/while/IdentityIdentitylstm_23/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_23_while_identitylstm_23/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
╦
Ђ
H__inference_lstm_cell_21_layer_call_and_return_conditional_losses_680126

inputs

states
states_10
matmul_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         Љ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_namestates:OK
'
_output_shapes
:         
 
_user_specified_namestates
ј8
ѓ
C__inference_lstm_23_layer_call_and_return_conditional_losses_680763

inputs%
lstm_cell_23_680681:@%
lstm_cell_23_680683:@!
lstm_cell_23_680685:@
identityѕб$lstm_cell_23/StatefulPartitionedCallбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЭ
$lstm_cell_23/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_23_680681lstm_cell_23_680683lstm_cell_23_680685*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_lstm_cell_23_layer_call_and_return_conditional_losses_680680n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : и
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_23_680681lstm_cell_23_680683lstm_cell_23_680685*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_680694*
condR
while_cond_680693*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         u
NoOpNoOp%^lstm_cell_23/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2L
$lstm_cell_23/StatefulPartitionedCall$lstm_cell_23/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
ИA
╩

lstm_22_while_body_682834,
(lstm_22_while_lstm_22_while_loop_counter2
.lstm_22_while_lstm_22_while_maximum_iterations
lstm_22_while_placeholder
lstm_22_while_placeholder_1
lstm_22_while_placeholder_2
lstm_22_while_placeholder_3+
'lstm_22_while_lstm_22_strided_slice_1_0g
clstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0M
;lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0:@O
=lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0:@J
<lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0:@
lstm_22_while_identity
lstm_22_while_identity_1
lstm_22_while_identity_2
lstm_22_while_identity_3
lstm_22_while_identity_4
lstm_22_while_identity_5)
%lstm_22_while_lstm_22_strided_slice_1e
alstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensorK
9lstm_22_while_lstm_cell_22_matmul_readvariableop_resource:@M
;lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource:@H
:lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource:@ѕб1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOpб0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOpб2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpљ
?lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╬
1lstm_22/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0lstm_22_while_placeholderHlstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0г
0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp;lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0Л
!lstm_22/while/lstm_cell_22/MatMulMatMul8lstm_22/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @░
2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp=lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0И
#lstm_22/while/lstm_cell_22/MatMul_1MatMullstm_22_while_placeholder_2:lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @х
lstm_22/while/lstm_cell_22/addAddV2+lstm_22/while/lstm_cell_22/MatMul:product:0-lstm_22/while/lstm_cell_22/MatMul_1:product:0*
T0*'
_output_shapes
:         @ф
1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp<lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Й
"lstm_22/while/lstm_cell_22/BiasAddBiasAdd"lstm_22/while/lstm_cell_22/add:z:09lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @l
*lstm_22/while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Є
 lstm_22/while/lstm_cell_22/splitSplit3lstm_22/while/lstm_cell_22/split/split_dim:output:0+lstm_22/while/lstm_cell_22/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitі
"lstm_22/while/lstm_cell_22/SigmoidSigmoid)lstm_22/while/lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         ї
$lstm_22/while/lstm_cell_22/Sigmoid_1Sigmoid)lstm_22/while/lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         ъ
lstm_22/while/lstm_cell_22/mulMul(lstm_22/while/lstm_cell_22/Sigmoid_1:y:0lstm_22_while_placeholder_3*
T0*'
_output_shapes
:         ё
lstm_22/while/lstm_cell_22/ReluRelu)lstm_22/while/lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         ░
 lstm_22/while/lstm_cell_22/mul_1Mul&lstm_22/while/lstm_cell_22/Sigmoid:y:0-lstm_22/while/lstm_cell_22/Relu:activations:0*
T0*'
_output_shapes
:         Ц
 lstm_22/while/lstm_cell_22/add_1AddV2"lstm_22/while/lstm_cell_22/mul:z:0$lstm_22/while/lstm_cell_22/mul_1:z:0*
T0*'
_output_shapes
:         ї
$lstm_22/while/lstm_cell_22/Sigmoid_2Sigmoid)lstm_22/while/lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         Ђ
!lstm_22/while/lstm_cell_22/Relu_1Relu$lstm_22/while/lstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         ┤
 lstm_22/while/lstm_cell_22/mul_2Mul(lstm_22/while/lstm_cell_22/Sigmoid_2:y:0/lstm_22/while/lstm_cell_22/Relu_1:activations:0*
T0*'
_output_shapes
:         т
2lstm_22/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_22_while_placeholder_1lstm_22_while_placeholder$lstm_22/while/lstm_cell_22/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмU
lstm_22/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_22/while/addAddV2lstm_22_while_placeholderlstm_22/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_22/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Є
lstm_22/while/add_1AddV2(lstm_22_while_lstm_22_while_loop_counterlstm_22/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_22/while/IdentityIdentitylstm_22/while/add_1:z:0^lstm_22/while/NoOp*
T0*
_output_shapes
: і
lstm_22/while/Identity_1Identity.lstm_22_while_lstm_22_while_maximum_iterations^lstm_22/while/NoOp*
T0*
_output_shapes
: q
lstm_22/while/Identity_2Identitylstm_22/while/add:z:0^lstm_22/while/NoOp*
T0*
_output_shapes
: ъ
lstm_22/while/Identity_3IdentityBlstm_22/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_22/while/NoOp*
T0*
_output_shapes
: Љ
lstm_22/while/Identity_4Identity$lstm_22/while/lstm_cell_22/mul_2:z:0^lstm_22/while/NoOp*
T0*'
_output_shapes
:         Љ
lstm_22/while/Identity_5Identity$lstm_22/while/lstm_cell_22/add_1:z:0^lstm_22/while/NoOp*
T0*'
_output_shapes
:         ­
lstm_22/while/NoOpNoOp2^lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp1^lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp3^lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_22_while_identitylstm_22/while/Identity:output:0"=
lstm_22_while_identity_1!lstm_22/while/Identity_1:output:0"=
lstm_22_while_identity_2!lstm_22/while/Identity_2:output:0"=
lstm_22_while_identity_3!lstm_22/while/Identity_3:output:0"=
lstm_22_while_identity_4!lstm_22/while/Identity_4:output:0"=
lstm_22_while_identity_5!lstm_22/while/Identity_5:output:0"P
%lstm_22_while_lstm_22_strided_slice_1'lstm_22_while_lstm_22_strided_slice_1_0"z
:lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource<lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0"|
;lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource=lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0"x
9lstm_22_while_lstm_cell_22_matmul_readvariableop_resource;lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0"╚
alstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensorclstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2f
1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp2d
0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp2h
2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
ЯP
в
&sequential_7_lstm_23_while_body_679798F
Bsequential_7_lstm_23_while_sequential_7_lstm_23_while_loop_counterL
Hsequential_7_lstm_23_while_sequential_7_lstm_23_while_maximum_iterations*
&sequential_7_lstm_23_while_placeholder,
(sequential_7_lstm_23_while_placeholder_1,
(sequential_7_lstm_23_while_placeholder_2,
(sequential_7_lstm_23_while_placeholder_3E
Asequential_7_lstm_23_while_sequential_7_lstm_23_strided_slice_1_0Ђ
}sequential_7_lstm_23_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_23_tensorarrayunstack_tensorlistfromtensor_0Z
Hsequential_7_lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0:@\
Jsequential_7_lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0:@W
Isequential_7_lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0:@'
#sequential_7_lstm_23_while_identity)
%sequential_7_lstm_23_while_identity_1)
%sequential_7_lstm_23_while_identity_2)
%sequential_7_lstm_23_while_identity_3)
%sequential_7_lstm_23_while_identity_4)
%sequential_7_lstm_23_while_identity_5C
?sequential_7_lstm_23_while_sequential_7_lstm_23_strided_slice_1
{sequential_7_lstm_23_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_23_tensorarrayunstack_tensorlistfromtensorX
Fsequential_7_lstm_23_while_lstm_cell_23_matmul_readvariableop_resource:@Z
Hsequential_7_lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource:@U
Gsequential_7_lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource:@ѕб>sequential_7/lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOpб=sequential_7/lstm_23/while/lstm_cell_23/MatMul/ReadVariableOpб?sequential_7/lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOpЮ
Lsequential_7/lstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Ј
>sequential_7/lstm_23/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_7_lstm_23_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_23_tensorarrayunstack_tensorlistfromtensor_0&sequential_7_lstm_23_while_placeholderUsequential_7/lstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0к
=sequential_7/lstm_23/while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOpHsequential_7_lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0Э
.sequential_7/lstm_23/while/lstm_cell_23/MatMulMatMulEsequential_7/lstm_23/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_7/lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @╩
?sequential_7/lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOpJsequential_7_lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0▀
0sequential_7/lstm_23/while/lstm_cell_23/MatMul_1MatMul(sequential_7_lstm_23_while_placeholder_2Gsequential_7/lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @▄
+sequential_7/lstm_23/while/lstm_cell_23/addAddV28sequential_7/lstm_23/while/lstm_cell_23/MatMul:product:0:sequential_7/lstm_23/while/lstm_cell_23/MatMul_1:product:0*
T0*'
_output_shapes
:         @─
>sequential_7/lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOpIsequential_7_lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0т
/sequential_7/lstm_23/while/lstm_cell_23/BiasAddBiasAdd/sequential_7/lstm_23/while/lstm_cell_23/add:z:0Fsequential_7/lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @y
7sequential_7/lstm_23/while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :«
-sequential_7/lstm_23/while/lstm_cell_23/splitSplit@sequential_7/lstm_23/while/lstm_cell_23/split/split_dim:output:08sequential_7/lstm_23/while/lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitц
/sequential_7/lstm_23/while/lstm_cell_23/SigmoidSigmoid6sequential_7/lstm_23/while/lstm_cell_23/split:output:0*
T0*'
_output_shapes
:         д
1sequential_7/lstm_23/while/lstm_cell_23/Sigmoid_1Sigmoid6sequential_7/lstm_23/while/lstm_cell_23/split:output:1*
T0*'
_output_shapes
:         ┼
+sequential_7/lstm_23/while/lstm_cell_23/mulMul5sequential_7/lstm_23/while/lstm_cell_23/Sigmoid_1:y:0(sequential_7_lstm_23_while_placeholder_3*
T0*'
_output_shapes
:         ъ
,sequential_7/lstm_23/while/lstm_cell_23/ReluRelu6sequential_7/lstm_23/while/lstm_cell_23/split:output:2*
T0*'
_output_shapes
:         О
-sequential_7/lstm_23/while/lstm_cell_23/mul_1Mul3sequential_7/lstm_23/while/lstm_cell_23/Sigmoid:y:0:sequential_7/lstm_23/while/lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         ╠
-sequential_7/lstm_23/while/lstm_cell_23/add_1AddV2/sequential_7/lstm_23/while/lstm_cell_23/mul:z:01sequential_7/lstm_23/while/lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         д
1sequential_7/lstm_23/while/lstm_cell_23/Sigmoid_2Sigmoid6sequential_7/lstm_23/while/lstm_cell_23/split:output:3*
T0*'
_output_shapes
:         Џ
.sequential_7/lstm_23/while/lstm_cell_23/Relu_1Relu1sequential_7/lstm_23/while/lstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         █
-sequential_7/lstm_23/while/lstm_cell_23/mul_2Mul5sequential_7/lstm_23/while/lstm_cell_23/Sigmoid_2:y:0<sequential_7/lstm_23/while/lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         Ў
?sequential_7/lstm_23/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_7_lstm_23_while_placeholder_1&sequential_7_lstm_23_while_placeholder1sequential_7/lstm_23/while/lstm_cell_23/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмb
 sequential_7/lstm_23/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Џ
sequential_7/lstm_23/while/addAddV2&sequential_7_lstm_23_while_placeholder)sequential_7/lstm_23/while/add/y:output:0*
T0*
_output_shapes
: d
"sequential_7/lstm_23/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :╗
 sequential_7/lstm_23/while/add_1AddV2Bsequential_7_lstm_23_while_sequential_7_lstm_23_while_loop_counter+sequential_7/lstm_23/while/add_1/y:output:0*
T0*
_output_shapes
: ў
#sequential_7/lstm_23/while/IdentityIdentity$sequential_7/lstm_23/while/add_1:z:0 ^sequential_7/lstm_23/while/NoOp*
T0*
_output_shapes
: Й
%sequential_7/lstm_23/while/Identity_1IdentityHsequential_7_lstm_23_while_sequential_7_lstm_23_while_maximum_iterations ^sequential_7/lstm_23/while/NoOp*
T0*
_output_shapes
: ў
%sequential_7/lstm_23/while/Identity_2Identity"sequential_7/lstm_23/while/add:z:0 ^sequential_7/lstm_23/while/NoOp*
T0*
_output_shapes
: ┼
%sequential_7/lstm_23/while/Identity_3IdentityOsequential_7/lstm_23/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_7/lstm_23/while/NoOp*
T0*
_output_shapes
: И
%sequential_7/lstm_23/while/Identity_4Identity1sequential_7/lstm_23/while/lstm_cell_23/mul_2:z:0 ^sequential_7/lstm_23/while/NoOp*
T0*'
_output_shapes
:         И
%sequential_7/lstm_23/while/Identity_5Identity1sequential_7/lstm_23/while/lstm_cell_23/add_1:z:0 ^sequential_7/lstm_23/while/NoOp*
T0*'
_output_shapes
:         ц
sequential_7/lstm_23/while/NoOpNoOp?^sequential_7/lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp>^sequential_7/lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp@^sequential_7/lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "S
#sequential_7_lstm_23_while_identity,sequential_7/lstm_23/while/Identity:output:0"W
%sequential_7_lstm_23_while_identity_1.sequential_7/lstm_23/while/Identity_1:output:0"W
%sequential_7_lstm_23_while_identity_2.sequential_7/lstm_23/while/Identity_2:output:0"W
%sequential_7_lstm_23_while_identity_3.sequential_7/lstm_23/while/Identity_3:output:0"W
%sequential_7_lstm_23_while_identity_4.sequential_7/lstm_23/while/Identity_4:output:0"W
%sequential_7_lstm_23_while_identity_5.sequential_7/lstm_23/while/Identity_5:output:0"ћ
Gsequential_7_lstm_23_while_lstm_cell_23_biasadd_readvariableop_resourceIsequential_7_lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0"ќ
Hsequential_7_lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resourceJsequential_7_lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0"њ
Fsequential_7_lstm_23_while_lstm_cell_23_matmul_readvariableop_resourceHsequential_7_lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0"ё
?sequential_7_lstm_23_while_sequential_7_lstm_23_strided_slice_1Asequential_7_lstm_23_while_sequential_7_lstm_23_strided_slice_1_0"Ч
{sequential_7_lstm_23_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_23_tensorarrayunstack_tensorlistfromtensor}sequential_7_lstm_23_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_23_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2ђ
>sequential_7/lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp>sequential_7/lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp2~
=sequential_7/lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp=sequential_7/lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp2ѓ
?sequential_7/lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp?sequential_7/lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
с"
П
while_body_680885
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_23_680909_0:@-
while_lstm_cell_23_680911_0:@)
while_lstm_cell_23_680913_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_23_680909:@+
while_lstm_cell_23_680911:@'
while_lstm_cell_23_680913:@ѕб*while/lstm_cell_23/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Х
*while/lstm_cell_23/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_23_680909_0while_lstm_cell_23_680911_0while_lstm_cell_23_680913_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_lstm_cell_23_layer_call_and_return_conditional_losses_680826▄
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_23/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: љ
while/Identity_4Identity3while/lstm_cell_23/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         љ
while/Identity_5Identity3while/lstm_cell_23/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         y

while/NoOpNoOp+^while/lstm_cell_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_23_680909while_lstm_cell_23_680909_0"8
while_lstm_cell_23_680911while_lstm_cell_23_680911_0"8
while_lstm_cell_23_680913while_lstm_cell_23_680913_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2X
*while/lstm_cell_23/StatefulPartitionedCall*while/lstm_cell_23/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
З	
e
F__inference_dropout_35_layer_call_and_return_conditional_losses_681696

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
є
d
+__inference_dropout_34_layer_call_fn_684854

inputs
identityѕбStatefulPartitionedCall╚
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_34_layer_call_and_return_conditional_losses_681884s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ц
G
+__inference_dropout_35_layer_call_fn_685492

inputs
identity┤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_35_layer_call_and_return_conditional_losses_681439`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ќ

с
lstm_22_while_cond_683294,
(lstm_22_while_lstm_22_while_loop_counter2
.lstm_22_while_lstm_22_while_maximum_iterations
lstm_22_while_placeholder
lstm_22_while_placeholder_1
lstm_22_while_placeholder_2
lstm_22_while_placeholder_3.
*lstm_22_while_less_lstm_22_strided_slice_1D
@lstm_22_while_lstm_22_while_cond_683294___redundant_placeholder0D
@lstm_22_while_lstm_22_while_cond_683294___redundant_placeholder1D
@lstm_22_while_lstm_22_while_cond_683294___redundant_placeholder2D
@lstm_22_while_lstm_22_while_cond_683294___redundant_placeholder3
lstm_22_while_identity
ѓ
lstm_22/while/LessLesslstm_22_while_placeholder*lstm_22_while_less_lstm_22_strided_slice_1*
T0*
_output_shapes
: [
lstm_22/while/IdentityIdentitylstm_22/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_22_while_identitylstm_22/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
с"
П
while_body_680185
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_21_680209_0:@-
while_lstm_cell_21_680211_0:@)
while_lstm_cell_21_680213_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_21_680209:@+
while_lstm_cell_21_680211:@'
while_lstm_cell_21_680213:@ѕб*while/lstm_cell_21/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Х
*while/lstm_cell_21/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_21_680209_0while_lstm_cell_21_680211_0while_lstm_cell_21_680213_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_lstm_cell_21_layer_call_and_return_conditional_losses_680126▄
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_21/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: љ
while/Identity_4Identity3while/lstm_cell_21/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         љ
while/Identity_5Identity3while/lstm_cell_21/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         y

while/NoOpNoOp+^while/lstm_cell_21/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_21_680209while_lstm_cell_21_680209_0"8
while_lstm_cell_21_680211while_lstm_cell_21_680211_0"8
while_lstm_cell_21_680213while_lstm_cell_21_680213_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2X
*while/lstm_cell_21/StatefulPartitionedCall*while/lstm_cell_21/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
ЎJ
ў
C__inference_lstm_22_layer_call_and_return_conditional_losses_681269

inputs=
+lstm_cell_22_matmul_readvariableop_resource:@?
-lstm_cell_22_matmul_1_readvariableop_resource:@:
,lstm_cell_22_biasadd_readvariableop_resource:@
identityѕб#lstm_cell_22/BiasAdd/ReadVariableOpб"lstm_cell_22/MatMul/ReadVariableOpб$lstm_cell_22/MatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskј
"lstm_cell_22/MatMul/ReadVariableOpReadVariableOp+lstm_cell_22_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ћ
lstm_cell_22/MatMulMatMulstrided_slice_2:output:0*lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @њ
$lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_22_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell_22/MatMul_1MatMulzeros:output:0,lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @І
lstm_cell_22/addAddV2lstm_cell_22/MatMul:product:0lstm_cell_22/MatMul_1:product:0*
T0*'
_output_shapes
:         @ї
#lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ћ
lstm_cell_22/BiasAddBiasAddlstm_cell_22/add:z:0+lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @^
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0lstm_cell_22/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitn
lstm_cell_22/SigmoidSigmoidlstm_cell_22/split:output:0*
T0*'
_output_shapes
:         p
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/split:output:1*
T0*'
_output_shapes
:         w
lstm_cell_22/mulMullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         h
lstm_cell_22/ReluRelulstm_cell_22/split:output:2*
T0*'
_output_shapes
:         є
lstm_cell_22/mul_1Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Relu:activations:0*
T0*'
_output_shapes
:         {
lstm_cell_22/add_1AddV2lstm_cell_22/mul:z:0lstm_cell_22/mul_1:z:0*
T0*'
_output_shapes
:         p
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/split:output:3*
T0*'
_output_shapes
:         e
lstm_cell_22/Relu_1Relulstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         і
lstm_cell_22/mul_2Mullstm_cell_22/Sigmoid_2:y:0!lstm_cell_22/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѓ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_22_matmul_readvariableop_resource-lstm_cell_22_matmul_1_readvariableop_resource,lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_681185*
condR
while_cond_681184*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         └
NoOpNoOp$^lstm_cell_22/BiasAdd/ReadVariableOp#^lstm_cell_22/MatMul/ReadVariableOp%^lstm_cell_22/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2J
#lstm_cell_22/BiasAdd/ReadVariableOp#lstm_cell_22/BiasAdd/ReadVariableOp2H
"lstm_cell_22/MatMul/ReadVariableOp"lstm_cell_22/MatMul/ReadVariableOp2L
$lstm_cell_22/MatMul_1/ReadVariableOp$lstm_cell_22/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
 7
╩
while_body_681185
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_22_matmul_readvariableop_resource_0:@G
5while_lstm_cell_22_matmul_1_readvariableop_resource_0:@B
4while_lstm_cell_22_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_22_matmul_readvariableop_resource:@E
3while_lstm_cell_22_matmul_1_readvariableop_resource:@@
2while_lstm_cell_22_biasadd_readvariableop_resource:@ѕб)while/lstm_cell_22/BiasAdd/ReadVariableOpб(while/lstm_cell_22/MatMul/ReadVariableOpб*while/lstm_cell_22/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ю
(while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_22_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0╣
while/lstm_cell_22/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @а
*while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_22_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0а
while/lstm_cell_22/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_22/addAddV2#while/lstm_cell_22/MatMul:product:0%while/lstm_cell_22/MatMul_1:product:0*
T0*'
_output_shapes
:         @џ
)while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0д
while/lstm_cell_22/BiasAddBiasAddwhile/lstm_cell_22/add:z:01while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0#while/lstm_cell_22/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitz
while/lstm_cell_22/SigmoidSigmoid!while/lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         |
while/lstm_cell_22/Sigmoid_1Sigmoid!while/lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         є
while/lstm_cell_22/mulMul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         t
while/lstm_cell_22/ReluRelu!while/lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         ў
while/lstm_cell_22/mul_1Mulwhile/lstm_cell_22/Sigmoid:y:0%while/lstm_cell_22/Relu:activations:0*
T0*'
_output_shapes
:         Ї
while/lstm_cell_22/add_1AddV2while/lstm_cell_22/mul:z:0while/lstm_cell_22/mul_1:z:0*
T0*'
_output_shapes
:         |
while/lstm_cell_22/Sigmoid_2Sigmoid!while/lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         q
while/lstm_cell_22/Relu_1Reluwhile/lstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         ю
while/lstm_cell_22/mul_2Mul while/lstm_cell_22/Sigmoid_2:y:0'while/lstm_cell_22/Relu_1:activations:0*
T0*'
_output_shapes
:         ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_22/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_22/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         y
while/Identity_5Identitywhile/lstm_cell_22/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         л

while/NoOpNoOp*^while/lstm_cell_22/BiasAdd/ReadVariableOp)^while/lstm_cell_22/MatMul/ReadVariableOp+^while/lstm_cell_22/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_22_biasadd_readvariableop_resource4while_lstm_cell_22_biasadd_readvariableop_resource_0"l
3while_lstm_cell_22_matmul_1_readvariableop_resource5while_lstm_cell_22_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_22_matmul_readvariableop_resource3while_lstm_cell_22_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2V
)while/lstm_cell_22/BiasAdd/ReadVariableOp)while/lstm_cell_22/BiasAdd/ReadVariableOp2T
(while/lstm_cell_22/MatMul/ReadVariableOp(while/lstm_cell_22/MatMul/ReadVariableOp2X
*while/lstm_cell_22/MatMul_1/ReadVariableOp*while/lstm_cell_22/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
П
╗
-__inference_sequential_7_layer_call_fn_682635

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:
identityѕбStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *3
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_682332o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
 7
╩
while_body_683974
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_21_matmul_readvariableop_resource_0:@G
5while_lstm_cell_21_matmul_1_readvariableop_resource_0:@B
4while_lstm_cell_21_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_21_matmul_readvariableop_resource:@E
3while_lstm_cell_21_matmul_1_readvariableop_resource:@@
2while_lstm_cell_21_biasadd_readvariableop_resource:@ѕб)while/lstm_cell_21/BiasAdd/ReadVariableOpб(while/lstm_cell_21/MatMul/ReadVariableOpб*while/lstm_cell_21/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ю
(while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_21_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0╣
while/lstm_cell_21/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @а
*while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_21_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0а
while/lstm_cell_21/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_21/addAddV2#while/lstm_cell_21/MatMul:product:0%while/lstm_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:         @џ
)while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0д
while/lstm_cell_21/BiasAddBiasAddwhile/lstm_cell_21/add:z:01while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
"while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_21/splitSplit+while/lstm_cell_21/split/split_dim:output:0#while/lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitz
while/lstm_cell_21/SigmoidSigmoid!while/lstm_cell_21/split:output:0*
T0*'
_output_shapes
:         |
while/lstm_cell_21/Sigmoid_1Sigmoid!while/lstm_cell_21/split:output:1*
T0*'
_output_shapes
:         є
while/lstm_cell_21/mulMul while/lstm_cell_21/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         t
while/lstm_cell_21/ReluRelu!while/lstm_cell_21/split:output:2*
T0*'
_output_shapes
:         ў
while/lstm_cell_21/mul_1Mulwhile/lstm_cell_21/Sigmoid:y:0%while/lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         Ї
while/lstm_cell_21/add_1AddV2while/lstm_cell_21/mul:z:0while/lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         |
while/lstm_cell_21/Sigmoid_2Sigmoid!while/lstm_cell_21/split:output:3*
T0*'
_output_shapes
:         q
while/lstm_cell_21/Relu_1Reluwhile/lstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         ю
while/lstm_cell_21/mul_2Mul while/lstm_cell_21/Sigmoid_2:y:0'while/lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_21/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_21/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         y
while/Identity_5Identitywhile/lstm_cell_21/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         л

while/NoOpNoOp*^while/lstm_cell_21/BiasAdd/ReadVariableOp)^while/lstm_cell_21/MatMul/ReadVariableOp+^while/lstm_cell_21/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_21_biasadd_readvariableop_resource4while_lstm_cell_21_biasadd_readvariableop_resource_0"l
3while_lstm_cell_21_matmul_1_readvariableop_resource5while_lstm_cell_21_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_21_matmul_readvariableop_resource3while_lstm_cell_21_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2V
)while/lstm_cell_21/BiasAdd/ReadVariableOp)while/lstm_cell_21/BiasAdd/ReadVariableOp2T
(while/lstm_cell_21/MatMul/ReadVariableOp(while/lstm_cell_21/MatMul/ReadVariableOp2X
*while/lstm_cell_21/MatMul_1/ReadVariableOp*while/lstm_cell_21/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
х
├
while_cond_683687
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_683687___redundant_placeholder04
0while_while_cond_683687___redundant_placeholder14
0while_while_cond_683687___redundant_placeholder24
0while_while_cond_683687___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
є
d
+__inference_dropout_33_layer_call_fn_684211

inputs
identityѕбStatefulPartitionedCall╚
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_33_layer_call_and_return_conditional_losses_682072s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
К	
ш
D__inference_dense_22_layer_call_and_return_conditional_losses_685674

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
З	
e
F__inference_dropout_35_layer_call_and_return_conditional_losses_685514

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
х
├
while_cond_681027
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_681027___redundant_placeholder04
0while_while_cond_681027___redundant_placeholder14
0while_while_cond_681027___redundant_placeholder24
0while_while_cond_681027___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
┤
G
+__inference_dropout_33_layer_call_fn_684206

inputs
identityИ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_33_layer_call_and_return_conditional_losses_681125d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         :S O
+
_output_shapes
:         
 
_user_specified_nameinputs
МJ
џ
C__inference_lstm_23_layer_call_and_return_conditional_losses_685058
inputs_0=
+lstm_cell_23_matmul_readvariableop_resource:@?
-lstm_cell_23_matmul_1_readvariableop_resource:@:
,lstm_cell_23_biasadd_readvariableop_resource:@
identityѕб#lstm_cell_23/BiasAdd/ReadVariableOpб"lstm_cell_23/MatMul/ReadVariableOpб$lstm_cell_23/MatMul_1/ReadVariableOpбwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskј
"lstm_cell_23/MatMul/ReadVariableOpReadVariableOp+lstm_cell_23_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ћ
lstm_cell_23/MatMulMatMulstrided_slice_2:output:0*lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @њ
$lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_23_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell_23/MatMul_1MatMulzeros:output:0,lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @І
lstm_cell_23/addAddV2lstm_cell_23/MatMul:product:0lstm_cell_23/MatMul_1:product:0*
T0*'
_output_shapes
:         @ї
#lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ћ
lstm_cell_23/BiasAddBiasAddlstm_cell_23/add:z:0+lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @^
lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_23/splitSplit%lstm_cell_23/split/split_dim:output:0lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitn
lstm_cell_23/SigmoidSigmoidlstm_cell_23/split:output:0*
T0*'
_output_shapes
:         p
lstm_cell_23/Sigmoid_1Sigmoidlstm_cell_23/split:output:1*
T0*'
_output_shapes
:         w
lstm_cell_23/mulMullstm_cell_23/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         h
lstm_cell_23/ReluRelulstm_cell_23/split:output:2*
T0*'
_output_shapes
:         є
lstm_cell_23/mul_1Mullstm_cell_23/Sigmoid:y:0lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         {
lstm_cell_23/add_1AddV2lstm_cell_23/mul:z:0lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         p
lstm_cell_23/Sigmoid_2Sigmoidlstm_cell_23/split:output:3*
T0*'
_output_shapes
:         e
lstm_cell_23/Relu_1Relulstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         і
lstm_cell_23/mul_2Mullstm_cell_23/Sigmoid_2:y:0!lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѓ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_23_matmul_readvariableop_resource-lstm_cell_23_matmul_1_readvariableop_resource,lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_684974*
condR
while_cond_684973*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         └
NoOpNoOp$^lstm_cell_23/BiasAdd/ReadVariableOp#^lstm_cell_23/MatMul/ReadVariableOp%^lstm_cell_23/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2J
#lstm_cell_23/BiasAdd/ReadVariableOp#lstm_cell_23/BiasAdd/ReadVariableOp2H
"lstm_cell_23/MatMul/ReadVariableOp"lstm_cell_23/MatMul/ReadVariableOp2L
$lstm_cell_23/MatMul_1/ReadVariableOp$lstm_cell_23/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
в
з
-__inference_lstm_cell_22_layer_call_fn_685789

inputs
states_0
states_1
unknown:@
	unknown_0:@
	unknown_1:@
identity

identity_1

identity_2ѕбStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_680330o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
states/0:QM
'
_output_shapes
:         
"
_user_specified_name
states/1
э
▓
(__inference_lstm_23_layer_call_fn_684915

inputs
unknown:@
	unknown_0:@
	unknown_1:@
identityѕбStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_lstm_23_layer_call_and_return_conditional_losses_681855o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ћ

e
F__inference_dropout_33_layer_call_and_return_conditional_losses_682072

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:љ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=ф
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         ]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         :S O
+
_output_shapes
:         
 
_user_specified_nameinputs
 7
╩
while_body_681028
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_21_matmul_readvariableop_resource_0:@G
5while_lstm_cell_21_matmul_1_readvariableop_resource_0:@B
4while_lstm_cell_21_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_21_matmul_readvariableop_resource:@E
3while_lstm_cell_21_matmul_1_readvariableop_resource:@@
2while_lstm_cell_21_biasadd_readvariableop_resource:@ѕб)while/lstm_cell_21/BiasAdd/ReadVariableOpб(while/lstm_cell_21/MatMul/ReadVariableOpб*while/lstm_cell_21/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ю
(while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_21_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0╣
while/lstm_cell_21/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @а
*while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_21_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0а
while/lstm_cell_21/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_21/addAddV2#while/lstm_cell_21/MatMul:product:0%while/lstm_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:         @џ
)while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0д
while/lstm_cell_21/BiasAddBiasAddwhile/lstm_cell_21/add:z:01while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
"while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_21/splitSplit+while/lstm_cell_21/split/split_dim:output:0#while/lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitz
while/lstm_cell_21/SigmoidSigmoid!while/lstm_cell_21/split:output:0*
T0*'
_output_shapes
:         |
while/lstm_cell_21/Sigmoid_1Sigmoid!while/lstm_cell_21/split:output:1*
T0*'
_output_shapes
:         є
while/lstm_cell_21/mulMul while/lstm_cell_21/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         t
while/lstm_cell_21/ReluRelu!while/lstm_cell_21/split:output:2*
T0*'
_output_shapes
:         ў
while/lstm_cell_21/mul_1Mulwhile/lstm_cell_21/Sigmoid:y:0%while/lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         Ї
while/lstm_cell_21/add_1AddV2while/lstm_cell_21/mul:z:0while/lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         |
while/lstm_cell_21/Sigmoid_2Sigmoid!while/lstm_cell_21/split:output:3*
T0*'
_output_shapes
:         q
while/lstm_cell_21/Relu_1Reluwhile/lstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         ю
while/lstm_cell_21/mul_2Mul while/lstm_cell_21/Sigmoid_2:y:0'while/lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_21/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_21/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         y
while/Identity_5Identitywhile/lstm_cell_21/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         л

while/NoOpNoOp*^while/lstm_cell_21/BiasAdd/ReadVariableOp)^while/lstm_cell_21/MatMul/ReadVariableOp+^while/lstm_cell_21/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_21_biasadd_readvariableop_resource4while_lstm_cell_21_biasadd_readvariableop_resource_0"l
3while_lstm_cell_21_matmul_1_readvariableop_resource5while_lstm_cell_21_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_21_matmul_readvariableop_resource3while_lstm_cell_21_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2V
)while/lstm_cell_21/BiasAdd/ReadVariableOp)while/lstm_cell_21/BiasAdd/ReadVariableOp2T
(while/lstm_cell_21/MatMul/ReadVariableOp(while/lstm_cell_21/MatMul/ReadVariableOp2X
*while/lstm_cell_21/MatMul_1/ReadVariableOp*while/lstm_cell_21/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
Е
┤
(__inference_lstm_21_layer_call_fn_683607
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
identityѕбStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_lstm_21_layer_call_and_return_conditional_losses_680254|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
 
▓
(__inference_lstm_22_layer_call_fn_684272

inputs
unknown:@
	unknown_0:@
	unknown_1:@
identityѕбStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_682043s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ИA
╩

lstm_21_while_body_683148,
(lstm_21_while_lstm_21_while_loop_counter2
.lstm_21_while_lstm_21_while_maximum_iterations
lstm_21_while_placeholder
lstm_21_while_placeholder_1
lstm_21_while_placeholder_2
lstm_21_while_placeholder_3+
'lstm_21_while_lstm_21_strided_slice_1_0g
clstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensor_0M
;lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0:@O
=lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0:@J
<lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0:@
lstm_21_while_identity
lstm_21_while_identity_1
lstm_21_while_identity_2
lstm_21_while_identity_3
lstm_21_while_identity_4
lstm_21_while_identity_5)
%lstm_21_while_lstm_21_strided_slice_1e
alstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensorK
9lstm_21_while_lstm_cell_21_matmul_readvariableop_resource:@M
;lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource:@H
:lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource:@ѕб1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOpб0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOpб2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpљ
?lstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╬
1lstm_21/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensor_0lstm_21_while_placeholderHlstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0г
0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp;lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0Л
!lstm_21/while/lstm_cell_21/MatMulMatMul8lstm_21/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @░
2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp=lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0И
#lstm_21/while/lstm_cell_21/MatMul_1MatMullstm_21_while_placeholder_2:lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @х
lstm_21/while/lstm_cell_21/addAddV2+lstm_21/while/lstm_cell_21/MatMul:product:0-lstm_21/while/lstm_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:         @ф
1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp<lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Й
"lstm_21/while/lstm_cell_21/BiasAddBiasAdd"lstm_21/while/lstm_cell_21/add:z:09lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @l
*lstm_21/while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Є
 lstm_21/while/lstm_cell_21/splitSplit3lstm_21/while/lstm_cell_21/split/split_dim:output:0+lstm_21/while/lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitі
"lstm_21/while/lstm_cell_21/SigmoidSigmoid)lstm_21/while/lstm_cell_21/split:output:0*
T0*'
_output_shapes
:         ї
$lstm_21/while/lstm_cell_21/Sigmoid_1Sigmoid)lstm_21/while/lstm_cell_21/split:output:1*
T0*'
_output_shapes
:         ъ
lstm_21/while/lstm_cell_21/mulMul(lstm_21/while/lstm_cell_21/Sigmoid_1:y:0lstm_21_while_placeholder_3*
T0*'
_output_shapes
:         ё
lstm_21/while/lstm_cell_21/ReluRelu)lstm_21/while/lstm_cell_21/split:output:2*
T0*'
_output_shapes
:         ░
 lstm_21/while/lstm_cell_21/mul_1Mul&lstm_21/while/lstm_cell_21/Sigmoid:y:0-lstm_21/while/lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         Ц
 lstm_21/while/lstm_cell_21/add_1AddV2"lstm_21/while/lstm_cell_21/mul:z:0$lstm_21/while/lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         ї
$lstm_21/while/lstm_cell_21/Sigmoid_2Sigmoid)lstm_21/while/lstm_cell_21/split:output:3*
T0*'
_output_shapes
:         Ђ
!lstm_21/while/lstm_cell_21/Relu_1Relu$lstm_21/while/lstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         ┤
 lstm_21/while/lstm_cell_21/mul_2Mul(lstm_21/while/lstm_cell_21/Sigmoid_2:y:0/lstm_21/while/lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         т
2lstm_21/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_21_while_placeholder_1lstm_21_while_placeholder$lstm_21/while/lstm_cell_21/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмU
lstm_21/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_21/while/addAddV2lstm_21_while_placeholderlstm_21/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_21/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Є
lstm_21/while/add_1AddV2(lstm_21_while_lstm_21_while_loop_counterlstm_21/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_21/while/IdentityIdentitylstm_21/while/add_1:z:0^lstm_21/while/NoOp*
T0*
_output_shapes
: і
lstm_21/while/Identity_1Identity.lstm_21_while_lstm_21_while_maximum_iterations^lstm_21/while/NoOp*
T0*
_output_shapes
: q
lstm_21/while/Identity_2Identitylstm_21/while/add:z:0^lstm_21/while/NoOp*
T0*
_output_shapes
: ъ
lstm_21/while/Identity_3IdentityBlstm_21/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_21/while/NoOp*
T0*
_output_shapes
: Љ
lstm_21/while/Identity_4Identity$lstm_21/while/lstm_cell_21/mul_2:z:0^lstm_21/while/NoOp*
T0*'
_output_shapes
:         Љ
lstm_21/while/Identity_5Identity$lstm_21/while/lstm_cell_21/add_1:z:0^lstm_21/while/NoOp*
T0*'
_output_shapes
:         ­
lstm_21/while/NoOpNoOp2^lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp1^lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp3^lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_21_while_identitylstm_21/while/Identity:output:0"=
lstm_21_while_identity_1!lstm_21/while/Identity_1:output:0"=
lstm_21_while_identity_2!lstm_21/while/Identity_2:output:0"=
lstm_21_while_identity_3!lstm_21/while/Identity_3:output:0"=
lstm_21_while_identity_4!lstm_21/while/Identity_4:output:0"=
lstm_21_while_identity_5!lstm_21/while/Identity_5:output:0"P
%lstm_21_while_lstm_21_strided_slice_1'lstm_21_while_lstm_21_strided_slice_1_0"z
:lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource<lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0"|
;lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource=lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0"x
9lstm_21_while_lstm_cell_21_matmul_readvariableop_resource;lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0"╚
alstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensorclstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2f
1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp2d
0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp2h
2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
ОJ
џ
C__inference_lstm_22_layer_call_and_return_conditional_losses_684558
inputs_0=
+lstm_cell_22_matmul_readvariableop_resource:@?
-lstm_cell_22_matmul_1_readvariableop_resource:@:
,lstm_cell_22_biasadd_readvariableop_resource:@
identityѕб#lstm_cell_22/BiasAdd/ReadVariableOpб"lstm_cell_22/MatMul/ReadVariableOpб$lstm_cell_22/MatMul_1/ReadVariableOpбwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskј
"lstm_cell_22/MatMul/ReadVariableOpReadVariableOp+lstm_cell_22_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ћ
lstm_cell_22/MatMulMatMulstrided_slice_2:output:0*lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @њ
$lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_22_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell_22/MatMul_1MatMulzeros:output:0,lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @І
lstm_cell_22/addAddV2lstm_cell_22/MatMul:product:0lstm_cell_22/MatMul_1:product:0*
T0*'
_output_shapes
:         @ї
#lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ћ
lstm_cell_22/BiasAddBiasAddlstm_cell_22/add:z:0+lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @^
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0lstm_cell_22/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitn
lstm_cell_22/SigmoidSigmoidlstm_cell_22/split:output:0*
T0*'
_output_shapes
:         p
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/split:output:1*
T0*'
_output_shapes
:         w
lstm_cell_22/mulMullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         h
lstm_cell_22/ReluRelulstm_cell_22/split:output:2*
T0*'
_output_shapes
:         є
lstm_cell_22/mul_1Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Relu:activations:0*
T0*'
_output_shapes
:         {
lstm_cell_22/add_1AddV2lstm_cell_22/mul:z:0lstm_cell_22/mul_1:z:0*
T0*'
_output_shapes
:         p
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/split:output:3*
T0*'
_output_shapes
:         e
lstm_cell_22/Relu_1Relulstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         і
lstm_cell_22/mul_2Mullstm_cell_22/Sigmoid_2:y:0!lstm_cell_22/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѓ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_22_matmul_readvariableop_resource-lstm_cell_22_matmul_1_readvariableop_resource,lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_684474*
condR
while_cond_684473*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  └
NoOpNoOp$^lstm_cell_22/BiasAdd/ReadVariableOp#^lstm_cell_22/MatMul/ReadVariableOp%^lstm_cell_22/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2J
#lstm_cell_22/BiasAdd/ReadVariableOp#lstm_cell_22/BiasAdd/ReadVariableOp2H
"lstm_cell_22/MatMul/ReadVariableOp"lstm_cell_22/MatMul/ReadVariableOp2L
$lstm_cell_22/MatMul_1/ReadVariableOp$lstm_cell_22/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
х
├
while_cond_683973
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_683973___redundant_placeholder04
0while_while_cond_683973___redundant_placeholder14
0while_while_cond_683973___redundant_placeholder24
0while_while_cond_683973___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
Џ

ш
D__inference_dense_20_layer_call_and_return_conditional_losses_685581

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
њ8
ѓ
C__inference_lstm_22_layer_call_and_return_conditional_losses_680604

inputs%
lstm_cell_22_680522:@%
lstm_cell_22_680524:@!
lstm_cell_22_680526:@
identityѕб$lstm_cell_22/StatefulPartitionedCallбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЭ
$lstm_cell_22/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_22_680522lstm_cell_22_680524lstm_cell_22_680526*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_680476n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : и
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_22_680522lstm_cell_22_680524lstm_cell_22_680526*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_680535*
condR
while_cond_680534*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  u
NoOpNoOp%^lstm_cell_22/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2L
$lstm_cell_22/StatefulPartitionedCall$lstm_cell_22/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
ѓ
у
&sequential_7_lstm_21_while_cond_679517F
Bsequential_7_lstm_21_while_sequential_7_lstm_21_while_loop_counterL
Hsequential_7_lstm_21_while_sequential_7_lstm_21_while_maximum_iterations*
&sequential_7_lstm_21_while_placeholder,
(sequential_7_lstm_21_while_placeholder_1,
(sequential_7_lstm_21_while_placeholder_2,
(sequential_7_lstm_21_while_placeholder_3H
Dsequential_7_lstm_21_while_less_sequential_7_lstm_21_strided_slice_1^
Zsequential_7_lstm_21_while_sequential_7_lstm_21_while_cond_679517___redundant_placeholder0^
Zsequential_7_lstm_21_while_sequential_7_lstm_21_while_cond_679517___redundant_placeholder1^
Zsequential_7_lstm_21_while_sequential_7_lstm_21_while_cond_679517___redundant_placeholder2^
Zsequential_7_lstm_21_while_sequential_7_lstm_21_while_cond_679517___redundant_placeholder3'
#sequential_7_lstm_21_while_identity
Х
sequential_7/lstm_21/while/LessLess&sequential_7_lstm_21_while_placeholderDsequential_7_lstm_21_while_less_sequential_7_lstm_21_strided_slice_1*
T0*
_output_shapes
: u
#sequential_7/lstm_21/while/IdentityIdentity#sequential_7/lstm_21/while/Less:z:0*
T0
*
_output_shapes
: "S
#sequential_7_lstm_21_while_identity,sequential_7/lstm_21/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
ж
d
F__inference_dropout_33_layer_call_and_return_conditional_losses_681125

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         _

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         :S O
+
_output_shapes
:         
 
_user_specified_nameinputs
в
з
-__inference_lstm_cell_21_layer_call_fn_685708

inputs
states_0
states_1
unknown:@
	unknown_0:@
	unknown_1:@
identity

identity_1

identity_2ѕбStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_lstm_cell_21_layer_call_and_return_conditional_losses_680126o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
states/0:QM
'
_output_shapes
:         
"
_user_specified_name
states/1
М
Ѓ
H__inference_lstm_cell_21_layer_call_and_return_conditional_losses_685772

inputs
states_0
states_10
matmul_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         Љ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
states/0:QM
'
_output_shapes
:         
"
_user_specified_name
states/1
ц
G
+__inference_dropout_36_layer_call_fn_685539

inputs
identity┤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_681463`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
х
├
while_cond_684616
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_684616___redundant_placeholder04
0while_while_cond_684616___redundant_placeholder14
0while_while_cond_684616___redundant_placeholder24
0while_while_cond_684616___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
З	
e
F__inference_dropout_38_layer_call_and_return_conditional_losses_685655

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Е
┤
(__inference_lstm_22_layer_call_fn_684239
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
identityѕбStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_680413|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
Е
┤
(__inference_lstm_21_layer_call_fn_683596
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
identityѕбStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_lstm_21_layer_call_and_return_conditional_losses_680063|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
╦
Ђ
H__inference_lstm_cell_23_layer_call_and_return_conditional_losses_680680

inputs

states
states_10
matmul_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         Љ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_namestates:OK
'
_output_shapes
:         
 
_user_specified_namestates
Ш
d
+__inference_dropout_38_layer_call_fn_685638

inputs
identityѕбStatefulPartitionedCall─
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_681597o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┼
ќ
)__inference_dense_20_layer_call_fn_685570

inputs
unknown:
	unknown_0:
identityѕбStatefulPartitionedCall▄
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_681476o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
М
Ѓ
H__inference_lstm_cell_21_layer_call_and_return_conditional_losses_685740

inputs
states_0
states_10
matmul_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         Љ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
states/0:QM
'
_output_shapes
:         
"
_user_specified_name
states/1
 7
╩
while_body_685117
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_23_matmul_readvariableop_resource_0:@G
5while_lstm_cell_23_matmul_1_readvariableop_resource_0:@B
4while_lstm_cell_23_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_23_matmul_readvariableop_resource:@E
3while_lstm_cell_23_matmul_1_readvariableop_resource:@@
2while_lstm_cell_23_biasadd_readvariableop_resource:@ѕб)while/lstm_cell_23/BiasAdd/ReadVariableOpб(while/lstm_cell_23/MatMul/ReadVariableOpб*while/lstm_cell_23/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ю
(while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_23_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0╣
while/lstm_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @а
*while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_23_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0а
while/lstm_cell_23/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_23/addAddV2#while/lstm_cell_23/MatMul:product:0%while/lstm_cell_23/MatMul_1:product:0*
T0*'
_output_shapes
:         @џ
)while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0д
while/lstm_cell_23/BiasAddBiasAddwhile/lstm_cell_23/add:z:01while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
"while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_23/splitSplit+while/lstm_cell_23/split/split_dim:output:0#while/lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitz
while/lstm_cell_23/SigmoidSigmoid!while/lstm_cell_23/split:output:0*
T0*'
_output_shapes
:         |
while/lstm_cell_23/Sigmoid_1Sigmoid!while/lstm_cell_23/split:output:1*
T0*'
_output_shapes
:         є
while/lstm_cell_23/mulMul while/lstm_cell_23/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         t
while/lstm_cell_23/ReluRelu!while/lstm_cell_23/split:output:2*
T0*'
_output_shapes
:         ў
while/lstm_cell_23/mul_1Mulwhile/lstm_cell_23/Sigmoid:y:0%while/lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         Ї
while/lstm_cell_23/add_1AddV2while/lstm_cell_23/mul:z:0while/lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         |
while/lstm_cell_23/Sigmoid_2Sigmoid!while/lstm_cell_23/split:output:3*
T0*'
_output_shapes
:         q
while/lstm_cell_23/Relu_1Reluwhile/lstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         ю
while/lstm_cell_23/mul_2Mul while/lstm_cell_23/Sigmoid_2:y:0'while/lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_23/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_23/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         y
while/Identity_5Identitywhile/lstm_cell_23/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         л

while/NoOpNoOp*^while/lstm_cell_23/BiasAdd/ReadVariableOp)^while/lstm_cell_23/MatMul/ReadVariableOp+^while/lstm_cell_23/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_23_biasadd_readvariableop_resource4while_lstm_cell_23_biasadd_readvariableop_resource_0"l
3while_lstm_cell_23_matmul_1_readvariableop_resource5while_lstm_cell_23_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_23_matmul_readvariableop_resource3while_lstm_cell_23_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2V
)while/lstm_cell_23/BiasAdd/ReadVariableOp)while/lstm_cell_23/BiasAdd/ReadVariableOp2T
(while/lstm_cell_23/MatMul/ReadVariableOp(while/lstm_cell_23/MatMul/ReadVariableOp2X
*while/lstm_cell_23/MatMul_1/ReadVariableOp*while/lstm_cell_23/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
ћ

e
F__inference_dropout_34_layer_call_and_return_conditional_losses_681884

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:љ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=ф
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         ]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         :S O
+
_output_shapes
:         
 
_user_specified_nameinputs
 
▓
(__inference_lstm_21_layer_call_fn_683629

inputs
unknown:@
	unknown_0:@
	unknown_1:@
identityѕбStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_lstm_21_layer_call_and_return_conditional_losses_682231s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
я╚
Ѕ
H__inference_sequential_7_layer_call_and_return_conditional_losses_683585

inputsE
3lstm_21_lstm_cell_21_matmul_readvariableop_resource:@G
5lstm_21_lstm_cell_21_matmul_1_readvariableop_resource:@B
4lstm_21_lstm_cell_21_biasadd_readvariableop_resource:@E
3lstm_22_lstm_cell_22_matmul_readvariableop_resource:@G
5lstm_22_lstm_cell_22_matmul_1_readvariableop_resource:@B
4lstm_22_lstm_cell_22_biasadd_readvariableop_resource:@E
3lstm_23_lstm_cell_23_matmul_readvariableop_resource:@G
5lstm_23_lstm_cell_23_matmul_1_readvariableop_resource:@B
4lstm_23_lstm_cell_23_biasadd_readvariableop_resource:@9
'dense_19_matmul_readvariableop_resource:6
(dense_19_biasadd_readvariableop_resource:9
'dense_20_matmul_readvariableop_resource:6
(dense_20_biasadd_readvariableop_resource:9
'dense_21_matmul_readvariableop_resource:6
(dense_21_biasadd_readvariableop_resource:9
'dense_22_matmul_readvariableop_resource:6
(dense_22_biasadd_readvariableop_resource:
identityѕбdense_19/BiasAdd/ReadVariableOpбdense_19/MatMul/ReadVariableOpбdense_20/BiasAdd/ReadVariableOpбdense_20/MatMul/ReadVariableOpбdense_21/BiasAdd/ReadVariableOpбdense_21/MatMul/ReadVariableOpбdense_22/BiasAdd/ReadVariableOpбdense_22/MatMul/ReadVariableOpб+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOpб*lstm_21/lstm_cell_21/MatMul/ReadVariableOpб,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOpбlstm_21/whileб+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOpб*lstm_22/lstm_cell_22/MatMul/ReadVariableOpб,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOpбlstm_22/whileб+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOpб*lstm_23/lstm_cell_23/MatMul/ReadVariableOpб,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOpбlstm_23/whileC
lstm_21/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
lstm_21/strided_sliceStridedSlicelstm_21/Shape:output:0$lstm_21/strided_slice/stack:output:0&lstm_21/strided_slice/stack_1:output:0&lstm_21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_21/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :І
lstm_21/zeros/packedPacklstm_21/strided_slice:output:0lstm_21/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_21/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ё
lstm_21/zerosFilllstm_21/zeros/packed:output:0lstm_21/zeros/Const:output:0*
T0*'
_output_shapes
:         Z
lstm_21/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ј
lstm_21/zeros_1/packedPacklstm_21/strided_slice:output:0!lstm_21/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_21/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    і
lstm_21/zeros_1Filllstm_21/zeros_1/packed:output:0lstm_21/zeros_1/Const:output:0*
T0*'
_output_shapes
:         k
lstm_21/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_21/transpose	Transposeinputslstm_21/transpose/perm:output:0*
T0*+
_output_shapes
:         T
lstm_21/Shape_1Shapelstm_21/transpose:y:0*
T0*
_output_shapes
:g
lstm_21/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_21/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_21/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
lstm_21/strided_slice_1StridedSlicelstm_21/Shape_1:output:0&lstm_21/strided_slice_1/stack:output:0(lstm_21/strided_slice_1/stack_1:output:0(lstm_21/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_21/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╠
lstm_21/TensorArrayV2TensorListReserve,lstm_21/TensorArrayV2/element_shape:output:0 lstm_21/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмј
=lstm_21/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Э
/lstm_21/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_21/transpose:y:0Flstm_21/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмg
lstm_21/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_21/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_21/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Љ
lstm_21/strided_slice_2StridedSlicelstm_21/transpose:y:0&lstm_21/strided_slice_2/stack:output:0(lstm_21/strided_slice_2/stack_1:output:0(lstm_21/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskъ
*lstm_21/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3lstm_21_lstm_cell_21_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Г
lstm_21/lstm_cell_21/MatMulMatMul lstm_21/strided_slice_2:output:02lstm_21/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @б
,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5lstm_21_lstm_cell_21_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Д
lstm_21/lstm_cell_21/MatMul_1MatMullstm_21/zeros:output:04lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Б
lstm_21/lstm_cell_21/addAddV2%lstm_21/lstm_cell_21/MatMul:product:0'lstm_21/lstm_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:         @ю
+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4lstm_21_lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0г
lstm_21/lstm_cell_21/BiasAddBiasAddlstm_21/lstm_cell_21/add:z:03lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @f
$lstm_21/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ш
lstm_21/lstm_cell_21/splitSplit-lstm_21/lstm_cell_21/split/split_dim:output:0%lstm_21/lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_split~
lstm_21/lstm_cell_21/SigmoidSigmoid#lstm_21/lstm_cell_21/split:output:0*
T0*'
_output_shapes
:         ђ
lstm_21/lstm_cell_21/Sigmoid_1Sigmoid#lstm_21/lstm_cell_21/split:output:1*
T0*'
_output_shapes
:         Ј
lstm_21/lstm_cell_21/mulMul"lstm_21/lstm_cell_21/Sigmoid_1:y:0lstm_21/zeros_1:output:0*
T0*'
_output_shapes
:         x
lstm_21/lstm_cell_21/ReluRelu#lstm_21/lstm_cell_21/split:output:2*
T0*'
_output_shapes
:         ъ
lstm_21/lstm_cell_21/mul_1Mul lstm_21/lstm_cell_21/Sigmoid:y:0'lstm_21/lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         Њ
lstm_21/lstm_cell_21/add_1AddV2lstm_21/lstm_cell_21/mul:z:0lstm_21/lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         ђ
lstm_21/lstm_cell_21/Sigmoid_2Sigmoid#lstm_21/lstm_cell_21/split:output:3*
T0*'
_output_shapes
:         u
lstm_21/lstm_cell_21/Relu_1Relulstm_21/lstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         б
lstm_21/lstm_cell_21/mul_2Mul"lstm_21/lstm_cell_21/Sigmoid_2:y:0)lstm_21/lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         v
%lstm_21/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       л
lstm_21/TensorArrayV2_1TensorListReserve.lstm_21/TensorArrayV2_1/element_shape:output:0 lstm_21/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмN
lstm_21/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_21/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         \
lstm_21/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ы
lstm_21/whileWhile#lstm_21/while/loop_counter:output:0)lstm_21/while/maximum_iterations:output:0lstm_21/time:output:0 lstm_21/TensorArrayV2_1:handle:0lstm_21/zeros:output:0lstm_21/zeros_1:output:0 lstm_21/strided_slice_1:output:0?lstm_21/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_21_lstm_cell_21_matmul_readvariableop_resource5lstm_21_lstm_cell_21_matmul_1_readvariableop_resource4lstm_21_lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_21_while_body_683148*%
condR
lstm_21_while_cond_683147*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ѕ
8lstm_21/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┌
*lstm_21/TensorArrayV2Stack/TensorListStackTensorListStacklstm_21/while:output:3Alstm_21/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0p
lstm_21/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         i
lstm_21/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_21/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:»
lstm_21/strided_slice_3StridedSlice3lstm_21/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_21/strided_slice_3/stack:output:0(lstm_21/strided_slice_3/stack_1:output:0(lstm_21/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskm
lstm_21/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
lstm_21/transpose_1	Transpose3lstm_21/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_21/transpose_1/perm:output:0*
T0*+
_output_shapes
:         c
lstm_21/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ]
dropout_33/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Ј
dropout_33/dropout/MulMullstm_21/transpose_1:y:0!dropout_33/dropout/Const:output:0*
T0*+
_output_shapes
:         _
dropout_33/dropout/ShapeShapelstm_21/transpose_1:y:0*
T0*
_output_shapes
:д
/dropout_33/dropout/random_uniform/RandomUniformRandomUniform!dropout_33/dropout/Shape:output:0*
T0*+
_output_shapes
:         *
dtype0f
!dropout_33/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=╦
dropout_33/dropout/GreaterEqualGreaterEqual8dropout_33/dropout/random_uniform/RandomUniform:output:0*dropout_33/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         Ѕ
dropout_33/dropout/CastCast#dropout_33/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         ј
dropout_33/dropout/Mul_1Muldropout_33/dropout/Mul:z:0dropout_33/dropout/Cast:y:0*
T0*+
_output_shapes
:         Y
lstm_22/ShapeShapedropout_33/dropout/Mul_1:z:0*
T0*
_output_shapes
:e
lstm_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
lstm_22/strided_sliceStridedSlicelstm_22/Shape:output:0$lstm_22/strided_slice/stack:output:0&lstm_22/strided_slice/stack_1:output:0&lstm_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_22/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :І
lstm_22/zeros/packedPacklstm_22/strided_slice:output:0lstm_22/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_22/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ё
lstm_22/zerosFilllstm_22/zeros/packed:output:0lstm_22/zeros/Const:output:0*
T0*'
_output_shapes
:         Z
lstm_22/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ј
lstm_22/zeros_1/packedPacklstm_22/strided_slice:output:0!lstm_22/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_22/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    і
lstm_22/zeros_1Filllstm_22/zeros_1/packed:output:0lstm_22/zeros_1/Const:output:0*
T0*'
_output_shapes
:         k
lstm_22/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Њ
lstm_22/transpose	Transposedropout_33/dropout/Mul_1:z:0lstm_22/transpose/perm:output:0*
T0*+
_output_shapes
:         T
lstm_22/Shape_1Shapelstm_22/transpose:y:0*
T0*
_output_shapes
:g
lstm_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
lstm_22/strided_slice_1StridedSlicelstm_22/Shape_1:output:0&lstm_22/strided_slice_1/stack:output:0(lstm_22/strided_slice_1/stack_1:output:0(lstm_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_22/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╠
lstm_22/TensorArrayV2TensorListReserve,lstm_22/TensorArrayV2/element_shape:output:0 lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмј
=lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Э
/lstm_22/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_22/transpose:y:0Flstm_22/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмg
lstm_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Љ
lstm_22/strided_slice_2StridedSlicelstm_22/transpose:y:0&lstm_22/strided_slice_2/stack:output:0(lstm_22/strided_slice_2/stack_1:output:0(lstm_22/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskъ
*lstm_22/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3lstm_22_lstm_cell_22_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Г
lstm_22/lstm_cell_22/MatMulMatMul lstm_22/strided_slice_2:output:02lstm_22/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @б
,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5lstm_22_lstm_cell_22_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Д
lstm_22/lstm_cell_22/MatMul_1MatMullstm_22/zeros:output:04lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Б
lstm_22/lstm_cell_22/addAddV2%lstm_22/lstm_cell_22/MatMul:product:0'lstm_22/lstm_cell_22/MatMul_1:product:0*
T0*'
_output_shapes
:         @ю
+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4lstm_22_lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0г
lstm_22/lstm_cell_22/BiasAddBiasAddlstm_22/lstm_cell_22/add:z:03lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @f
$lstm_22/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ш
lstm_22/lstm_cell_22/splitSplit-lstm_22/lstm_cell_22/split/split_dim:output:0%lstm_22/lstm_cell_22/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_split~
lstm_22/lstm_cell_22/SigmoidSigmoid#lstm_22/lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         ђ
lstm_22/lstm_cell_22/Sigmoid_1Sigmoid#lstm_22/lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         Ј
lstm_22/lstm_cell_22/mulMul"lstm_22/lstm_cell_22/Sigmoid_1:y:0lstm_22/zeros_1:output:0*
T0*'
_output_shapes
:         x
lstm_22/lstm_cell_22/ReluRelu#lstm_22/lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         ъ
lstm_22/lstm_cell_22/mul_1Mul lstm_22/lstm_cell_22/Sigmoid:y:0'lstm_22/lstm_cell_22/Relu:activations:0*
T0*'
_output_shapes
:         Њ
lstm_22/lstm_cell_22/add_1AddV2lstm_22/lstm_cell_22/mul:z:0lstm_22/lstm_cell_22/mul_1:z:0*
T0*'
_output_shapes
:         ђ
lstm_22/lstm_cell_22/Sigmoid_2Sigmoid#lstm_22/lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         u
lstm_22/lstm_cell_22/Relu_1Relulstm_22/lstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         б
lstm_22/lstm_cell_22/mul_2Mul"lstm_22/lstm_cell_22/Sigmoid_2:y:0)lstm_22/lstm_cell_22/Relu_1:activations:0*
T0*'
_output_shapes
:         v
%lstm_22/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       л
lstm_22/TensorArrayV2_1TensorListReserve.lstm_22/TensorArrayV2_1/element_shape:output:0 lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмN
lstm_22/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_22/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         \
lstm_22/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ы
lstm_22/whileWhile#lstm_22/while/loop_counter:output:0)lstm_22/while/maximum_iterations:output:0lstm_22/time:output:0 lstm_22/TensorArrayV2_1:handle:0lstm_22/zeros:output:0lstm_22/zeros_1:output:0 lstm_22/strided_slice_1:output:0?lstm_22/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_22_lstm_cell_22_matmul_readvariableop_resource5lstm_22_lstm_cell_22_matmul_1_readvariableop_resource4lstm_22_lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_22_while_body_683295*%
condR
lstm_22_while_cond_683294*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ѕ
8lstm_22/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┌
*lstm_22/TensorArrayV2Stack/TensorListStackTensorListStacklstm_22/while:output:3Alstm_22/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0p
lstm_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         i
lstm_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:»
lstm_22/strided_slice_3StridedSlice3lstm_22/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_22/strided_slice_3/stack:output:0(lstm_22/strided_slice_3/stack_1:output:0(lstm_22/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskm
lstm_22/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
lstm_22/transpose_1	Transpose3lstm_22/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_22/transpose_1/perm:output:0*
T0*+
_output_shapes
:         c
lstm_22/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ]
dropout_34/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Ј
dropout_34/dropout/MulMullstm_22/transpose_1:y:0!dropout_34/dropout/Const:output:0*
T0*+
_output_shapes
:         _
dropout_34/dropout/ShapeShapelstm_22/transpose_1:y:0*
T0*
_output_shapes
:д
/dropout_34/dropout/random_uniform/RandomUniformRandomUniform!dropout_34/dropout/Shape:output:0*
T0*+
_output_shapes
:         *
dtype0f
!dropout_34/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=╦
dropout_34/dropout/GreaterEqualGreaterEqual8dropout_34/dropout/random_uniform/RandomUniform:output:0*dropout_34/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         Ѕ
dropout_34/dropout/CastCast#dropout_34/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         ј
dropout_34/dropout/Mul_1Muldropout_34/dropout/Mul:z:0dropout_34/dropout/Cast:y:0*
T0*+
_output_shapes
:         Y
lstm_23/ShapeShapedropout_34/dropout/Mul_1:z:0*
T0*
_output_shapes
:e
lstm_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
lstm_23/strided_sliceStridedSlicelstm_23/Shape:output:0$lstm_23/strided_slice/stack:output:0&lstm_23/strided_slice/stack_1:output:0&lstm_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_23/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :І
lstm_23/zeros/packedPacklstm_23/strided_slice:output:0lstm_23/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_23/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ё
lstm_23/zerosFilllstm_23/zeros/packed:output:0lstm_23/zeros/Const:output:0*
T0*'
_output_shapes
:         Z
lstm_23/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ј
lstm_23/zeros_1/packedPacklstm_23/strided_slice:output:0!lstm_23/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_23/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    і
lstm_23/zeros_1Filllstm_23/zeros_1/packed:output:0lstm_23/zeros_1/Const:output:0*
T0*'
_output_shapes
:         k
lstm_23/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Њ
lstm_23/transpose	Transposedropout_34/dropout/Mul_1:z:0lstm_23/transpose/perm:output:0*
T0*+
_output_shapes
:         T
lstm_23/Shape_1Shapelstm_23/transpose:y:0*
T0*
_output_shapes
:g
lstm_23/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_23/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_23/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
lstm_23/strided_slice_1StridedSlicelstm_23/Shape_1:output:0&lstm_23/strided_slice_1/stack:output:0(lstm_23/strided_slice_1/stack_1:output:0(lstm_23/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_23/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╠
lstm_23/TensorArrayV2TensorListReserve,lstm_23/TensorArrayV2/element_shape:output:0 lstm_23/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмј
=lstm_23/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Э
/lstm_23/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_23/transpose:y:0Flstm_23/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмg
lstm_23/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_23/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_23/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Љ
lstm_23/strided_slice_2StridedSlicelstm_23/transpose:y:0&lstm_23/strided_slice_2/stack:output:0(lstm_23/strided_slice_2/stack_1:output:0(lstm_23/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskъ
*lstm_23/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3lstm_23_lstm_cell_23_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Г
lstm_23/lstm_cell_23/MatMulMatMul lstm_23/strided_slice_2:output:02lstm_23/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @б
,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5lstm_23_lstm_cell_23_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Д
lstm_23/lstm_cell_23/MatMul_1MatMullstm_23/zeros:output:04lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Б
lstm_23/lstm_cell_23/addAddV2%lstm_23/lstm_cell_23/MatMul:product:0'lstm_23/lstm_cell_23/MatMul_1:product:0*
T0*'
_output_shapes
:         @ю
+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4lstm_23_lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0г
lstm_23/lstm_cell_23/BiasAddBiasAddlstm_23/lstm_cell_23/add:z:03lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @f
$lstm_23/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ш
lstm_23/lstm_cell_23/splitSplit-lstm_23/lstm_cell_23/split/split_dim:output:0%lstm_23/lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_split~
lstm_23/lstm_cell_23/SigmoidSigmoid#lstm_23/lstm_cell_23/split:output:0*
T0*'
_output_shapes
:         ђ
lstm_23/lstm_cell_23/Sigmoid_1Sigmoid#lstm_23/lstm_cell_23/split:output:1*
T0*'
_output_shapes
:         Ј
lstm_23/lstm_cell_23/mulMul"lstm_23/lstm_cell_23/Sigmoid_1:y:0lstm_23/zeros_1:output:0*
T0*'
_output_shapes
:         x
lstm_23/lstm_cell_23/ReluRelu#lstm_23/lstm_cell_23/split:output:2*
T0*'
_output_shapes
:         ъ
lstm_23/lstm_cell_23/mul_1Mul lstm_23/lstm_cell_23/Sigmoid:y:0'lstm_23/lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         Њ
lstm_23/lstm_cell_23/add_1AddV2lstm_23/lstm_cell_23/mul:z:0lstm_23/lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         ђ
lstm_23/lstm_cell_23/Sigmoid_2Sigmoid#lstm_23/lstm_cell_23/split:output:3*
T0*'
_output_shapes
:         u
lstm_23/lstm_cell_23/Relu_1Relulstm_23/lstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         б
lstm_23/lstm_cell_23/mul_2Mul"lstm_23/lstm_cell_23/Sigmoid_2:y:0)lstm_23/lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         v
%lstm_23/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       л
lstm_23/TensorArrayV2_1TensorListReserve.lstm_23/TensorArrayV2_1/element_shape:output:0 lstm_23/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмN
lstm_23/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_23/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         \
lstm_23/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ы
lstm_23/whileWhile#lstm_23/while/loop_counter:output:0)lstm_23/while/maximum_iterations:output:0lstm_23/time:output:0 lstm_23/TensorArrayV2_1:handle:0lstm_23/zeros:output:0lstm_23/zeros_1:output:0 lstm_23/strided_slice_1:output:0?lstm_23/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_23_lstm_cell_23_matmul_readvariableop_resource5lstm_23_lstm_cell_23_matmul_1_readvariableop_resource4lstm_23_lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_23_while_body_683442*%
condR
lstm_23_while_cond_683441*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ѕ
8lstm_23/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┌
*lstm_23/TensorArrayV2Stack/TensorListStackTensorListStacklstm_23/while:output:3Alstm_23/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0p
lstm_23/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         i
lstm_23/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_23/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:»
lstm_23/strided_slice_3StridedSlice3lstm_23/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_23/strided_slice_3/stack:output:0(lstm_23/strided_slice_3/stack_1:output:0(lstm_23/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskm
lstm_23/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
lstm_23/transpose_1	Transpose3lstm_23/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_23/transpose_1/perm:output:0*
T0*+
_output_shapes
:         c
lstm_23/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ]
dropout_35/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?ћ
dropout_35/dropout/MulMul lstm_23/strided_slice_3:output:0!dropout_35/dropout/Const:output:0*
T0*'
_output_shapes
:         h
dropout_35/dropout/ShapeShape lstm_23/strided_slice_3:output:0*
T0*
_output_shapes
:б
/dropout_35/dropout/random_uniform/RandomUniformRandomUniform!dropout_35/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0f
!dropout_35/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=К
dropout_35/dropout/GreaterEqualGreaterEqual8dropout_35/dropout/random_uniform/RandomUniform:output:0*dropout_35/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ё
dropout_35/dropout/CastCast#dropout_35/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         і
dropout_35/dropout/Mul_1Muldropout_35/dropout/Mul:z:0dropout_35/dropout/Cast:y:0*
T0*'
_output_shapes
:         є
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Љ
dense_19/MatMulMatMuldropout_35/dropout/Mul_1:z:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         b
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*'
_output_shapes
:         ]
dropout_36/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Ј
dropout_36/dropout/MulMuldense_19/Relu:activations:0!dropout_36/dropout/Const:output:0*
T0*'
_output_shapes
:         c
dropout_36/dropout/ShapeShapedense_19/Relu:activations:0*
T0*
_output_shapes
:б
/dropout_36/dropout/random_uniform/RandomUniformRandomUniform!dropout_36/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0f
!dropout_36/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=К
dropout_36/dropout/GreaterEqualGreaterEqual8dropout_36/dropout/random_uniform/RandomUniform:output:0*dropout_36/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ё
dropout_36/dropout/CastCast#dropout_36/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         і
dropout_36/dropout/Mul_1Muldropout_36/dropout/Mul:z:0dropout_36/dropout/Cast:y:0*
T0*'
_output_shapes
:         є
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Љ
dense_20/MatMulMatMuldropout_36/dropout/Mul_1:z:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         b
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*'
_output_shapes
:         ]
dropout_37/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Ј
dropout_37/dropout/MulMuldense_20/Relu:activations:0!dropout_37/dropout/Const:output:0*
T0*'
_output_shapes
:         c
dropout_37/dropout/ShapeShapedense_20/Relu:activations:0*
T0*
_output_shapes
:б
/dropout_37/dropout/random_uniform/RandomUniformRandomUniform!dropout_37/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0f
!dropout_37/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=К
dropout_37/dropout/GreaterEqualGreaterEqual8dropout_37/dropout/random_uniform/RandomUniform:output:0*dropout_37/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ё
dropout_37/dropout/CastCast#dropout_37/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         і
dropout_37/dropout/Mul_1Muldropout_37/dropout/Mul:z:0dropout_37/dropout/Cast:y:0*
T0*'
_output_shapes
:         є
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Љ
dense_21/MatMulMatMuldropout_37/dropout/Mul_1:z:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         b
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*'
_output_shapes
:         ]
dropout_38/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Ј
dropout_38/dropout/MulMuldense_21/Relu:activations:0!dropout_38/dropout/Const:output:0*
T0*'
_output_shapes
:         c
dropout_38/dropout/ShapeShapedense_21/Relu:activations:0*
T0*
_output_shapes
:б
/dropout_38/dropout/random_uniform/RandomUniformRandomUniform!dropout_38/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0f
!dropout_38/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=К
dropout_38/dropout/GreaterEqualGreaterEqual8dropout_38/dropout/random_uniform/RandomUniform:output:0*dropout_38/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ё
dropout_38/dropout/CastCast#dropout_38/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         і
dropout_38/dropout/Mul_1Muldropout_38/dropout/Mul:z:0dropout_38/dropout/Cast:y:0*
T0*'
_output_shapes
:         є
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Љ
dense_22/MatMulMatMuldropout_38/dropout/Mul_1:z:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
IdentityIdentitydense_22/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         а
NoOpNoOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp,^lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp+^lstm_21/lstm_cell_21/MatMul/ReadVariableOp-^lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp^lstm_21/while,^lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp+^lstm_22/lstm_cell_22/MatMul/ReadVariableOp-^lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp^lstm_22/while,^lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp+^lstm_23/lstm_cell_23/MatMul/ReadVariableOp-^lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp^lstm_23/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         : : : : : : : : : : : : : : : : : 2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2Z
+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp2X
*lstm_21/lstm_cell_21/MatMul/ReadVariableOp*lstm_21/lstm_cell_21/MatMul/ReadVariableOp2\
,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp2
lstm_21/whilelstm_21/while2Z
+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp2X
*lstm_22/lstm_cell_22/MatMul/ReadVariableOp*lstm_22/lstm_cell_22/MatMul/ReadVariableOp2\
,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp2
lstm_22/whilelstm_22/while2Z
+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp2X
*lstm_23/lstm_cell_23/MatMul/ReadVariableOp*lstm_23/lstm_cell_23/MatMul/ReadVariableOp2\
,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp2
lstm_23/whilelstm_23/while:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
х
├
while_cond_684473
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_684473___redundant_placeholder04
0while_while_cond_684473___redundant_placeholder14
0while_while_cond_684473___redundant_placeholder24
0while_while_cond_684473___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
Ј
┤
(__inference_lstm_23_layer_call_fn_684882
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
identityѕбStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_lstm_23_layer_call_and_return_conditional_losses_680763o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
 7
╩
while_body_684117
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_21_matmul_readvariableop_resource_0:@G
5while_lstm_cell_21_matmul_1_readvariableop_resource_0:@B
4while_lstm_cell_21_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_21_matmul_readvariableop_resource:@E
3while_lstm_cell_21_matmul_1_readvariableop_resource:@@
2while_lstm_cell_21_biasadd_readvariableop_resource:@ѕб)while/lstm_cell_21/BiasAdd/ReadVariableOpб(while/lstm_cell_21/MatMul/ReadVariableOpб*while/lstm_cell_21/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ю
(while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_21_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0╣
while/lstm_cell_21/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @а
*while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_21_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0а
while/lstm_cell_21/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_21/addAddV2#while/lstm_cell_21/MatMul:product:0%while/lstm_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:         @џ
)while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0д
while/lstm_cell_21/BiasAddBiasAddwhile/lstm_cell_21/add:z:01while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
"while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_21/splitSplit+while/lstm_cell_21/split/split_dim:output:0#while/lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitz
while/lstm_cell_21/SigmoidSigmoid!while/lstm_cell_21/split:output:0*
T0*'
_output_shapes
:         |
while/lstm_cell_21/Sigmoid_1Sigmoid!while/lstm_cell_21/split:output:1*
T0*'
_output_shapes
:         є
while/lstm_cell_21/mulMul while/lstm_cell_21/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         t
while/lstm_cell_21/ReluRelu!while/lstm_cell_21/split:output:2*
T0*'
_output_shapes
:         ў
while/lstm_cell_21/mul_1Mulwhile/lstm_cell_21/Sigmoid:y:0%while/lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         Ї
while/lstm_cell_21/add_1AddV2while/lstm_cell_21/mul:z:0while/lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         |
while/lstm_cell_21/Sigmoid_2Sigmoid!while/lstm_cell_21/split:output:3*
T0*'
_output_shapes
:         q
while/lstm_cell_21/Relu_1Reluwhile/lstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         ю
while/lstm_cell_21/mul_2Mul while/lstm_cell_21/Sigmoid_2:y:0'while/lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_21/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_21/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         y
while/Identity_5Identitywhile/lstm_cell_21/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         л

while/NoOpNoOp*^while/lstm_cell_21/BiasAdd/ReadVariableOp)^while/lstm_cell_21/MatMul/ReadVariableOp+^while/lstm_cell_21/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_21_biasadd_readvariableop_resource4while_lstm_cell_21_biasadd_readvariableop_resource_0"l
3while_lstm_cell_21_matmul_1_readvariableop_resource5while_lstm_cell_21_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_21_matmul_readvariableop_resource3while_lstm_cell_21_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2V
)while/lstm_cell_21/BiasAdd/ReadVariableOp)while/lstm_cell_21/BiasAdd/ReadVariableOp2T
(while/lstm_cell_21/MatMul/ReadVariableOp(while/lstm_cell_21/MatMul/ReadVariableOp2X
*while/lstm_cell_21/MatMul_1/ReadVariableOp*while/lstm_cell_21/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
с"
П
while_body_680344
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_22_680368_0:@-
while_lstm_cell_22_680370_0:@)
while_lstm_cell_22_680372_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_22_680368:@+
while_lstm_cell_22_680370:@'
while_lstm_cell_22_680372:@ѕб*while/lstm_cell_22/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Х
*while/lstm_cell_22/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_22_680368_0while_lstm_cell_22_680370_0while_lstm_cell_22_680372_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_680330▄
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_22/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: љ
while/Identity_4Identity3while/lstm_cell_22/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         љ
while/Identity_5Identity3while/lstm_cell_22/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         y

while/NoOpNoOp+^while/lstm_cell_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_22_680368while_lstm_cell_22_680368_0"8
while_lstm_cell_22_680370while_lstm_cell_22_680370_0"8
while_lstm_cell_22_680372while_lstm_cell_22_680372_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2X
*while/lstm_cell_22/StatefulPartitionedCall*while/lstm_cell_22/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
┘
d
F__inference_dropout_35_layer_call_and_return_conditional_losses_681439

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
 7
╩
while_body_684617
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_22_matmul_readvariableop_resource_0:@G
5while_lstm_cell_22_matmul_1_readvariableop_resource_0:@B
4while_lstm_cell_22_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_22_matmul_readvariableop_resource:@E
3while_lstm_cell_22_matmul_1_readvariableop_resource:@@
2while_lstm_cell_22_biasadd_readvariableop_resource:@ѕб)while/lstm_cell_22/BiasAdd/ReadVariableOpб(while/lstm_cell_22/MatMul/ReadVariableOpб*while/lstm_cell_22/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ю
(while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_22_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0╣
while/lstm_cell_22/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @а
*while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_22_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0а
while/lstm_cell_22/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_22/addAddV2#while/lstm_cell_22/MatMul:product:0%while/lstm_cell_22/MatMul_1:product:0*
T0*'
_output_shapes
:         @џ
)while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0д
while/lstm_cell_22/BiasAddBiasAddwhile/lstm_cell_22/add:z:01while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0#while/lstm_cell_22/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitz
while/lstm_cell_22/SigmoidSigmoid!while/lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         |
while/lstm_cell_22/Sigmoid_1Sigmoid!while/lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         є
while/lstm_cell_22/mulMul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         t
while/lstm_cell_22/ReluRelu!while/lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         ў
while/lstm_cell_22/mul_1Mulwhile/lstm_cell_22/Sigmoid:y:0%while/lstm_cell_22/Relu:activations:0*
T0*'
_output_shapes
:         Ї
while/lstm_cell_22/add_1AddV2while/lstm_cell_22/mul:z:0while/lstm_cell_22/mul_1:z:0*
T0*'
_output_shapes
:         |
while/lstm_cell_22/Sigmoid_2Sigmoid!while/lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         q
while/lstm_cell_22/Relu_1Reluwhile/lstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         ю
while/lstm_cell_22/mul_2Mul while/lstm_cell_22/Sigmoid_2:y:0'while/lstm_cell_22/Relu_1:activations:0*
T0*'
_output_shapes
:         ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_22/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_22/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         y
while/Identity_5Identitywhile/lstm_cell_22/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         л

while/NoOpNoOp*^while/lstm_cell_22/BiasAdd/ReadVariableOp)^while/lstm_cell_22/MatMul/ReadVariableOp+^while/lstm_cell_22/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_22_biasadd_readvariableop_resource4while_lstm_cell_22_biasadd_readvariableop_resource_0"l
3while_lstm_cell_22_matmul_1_readvariableop_resource5while_lstm_cell_22_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_22_matmul_readvariableop_resource3while_lstm_cell_22_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2V
)while/lstm_cell_22/BiasAdd/ReadVariableOp)while/lstm_cell_22/BiasAdd/ReadVariableOp2T
(while/lstm_cell_22/MatMul/ReadVariableOp(while/lstm_cell_22/MatMul/ReadVariableOp2X
*while/lstm_cell_22/MatMul_1/ReadVariableOp*while/lstm_cell_22/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
ї7
Є
H__inference_sequential_7_layer_call_and_return_conditional_losses_681530

inputs 
lstm_21_681113:@ 
lstm_21_681115:@
lstm_21_681117:@ 
lstm_22_681270:@ 
lstm_22_681272:@
lstm_22_681274:@ 
lstm_23_681427:@ 
lstm_23_681429:@
lstm_23_681431:@!
dense_19_681453:
dense_19_681455:!
dense_20_681477:
dense_20_681479:!
dense_21_681501:
dense_21_681503:!
dense_22_681524:
dense_22_681526:
identityѕб dense_19/StatefulPartitionedCallб dense_20/StatefulPartitionedCallб dense_21/StatefulPartitionedCallб dense_22/StatefulPartitionedCallбlstm_21/StatefulPartitionedCallбlstm_22/StatefulPartitionedCallбlstm_23/StatefulPartitionedCallЁ
lstm_21/StatefulPartitionedCallStatefulPartitionedCallinputslstm_21_681113lstm_21_681115lstm_21_681117*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_lstm_21_layer_call_and_return_conditional_losses_681112т
dropout_33/PartitionedCallPartitionedCall(lstm_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_33_layer_call_and_return_conditional_losses_681125б
lstm_22/StatefulPartitionedCallStatefulPartitionedCall#dropout_33/PartitionedCall:output:0lstm_22_681270lstm_22_681272lstm_22_681274*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_681269т
dropout_34/PartitionedCallPartitionedCall(lstm_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_34_layer_call_and_return_conditional_losses_681282ъ
lstm_23/StatefulPartitionedCallStatefulPartitionedCall#dropout_34/PartitionedCall:output:0lstm_23_681427lstm_23_681429lstm_23_681431*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_lstm_23_layer_call_and_return_conditional_losses_681426р
dropout_35/PartitionedCallPartitionedCall(lstm_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_35_layer_call_and_return_conditional_losses_681439љ
 dense_19/StatefulPartitionedCallStatefulPartitionedCall#dropout_35/PartitionedCall:output:0dense_19_681453dense_19_681455*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_681452Р
dropout_36/PartitionedCallPartitionedCall)dense_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_36_layer_call_and_return_conditional_losses_681463љ
 dense_20/StatefulPartitionedCallStatefulPartitionedCall#dropout_36/PartitionedCall:output:0dense_20_681477dense_20_681479*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_681476Р
dropout_37/PartitionedCallPartitionedCall)dense_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_681487љ
 dense_21/StatefulPartitionedCallStatefulPartitionedCall#dropout_37/PartitionedCall:output:0dense_21_681501dense_21_681503*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_681500Р
dropout_38/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_681511љ
 dense_22/StatefulPartitionedCallStatefulPartitionedCall#dropout_38/PartitionedCall:output:0dense_22_681524dense_22_681526*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_681523x
IdentityIdentity)dense_22/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         И
NoOpNoOp!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall ^lstm_21/StatefulPartitionedCall ^lstm_22/StatefulPartitionedCall ^lstm_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         : : : : : : : : : : : : : : : : : 2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2B
lstm_21/StatefulPartitionedCalllstm_21/StatefulPartitionedCall2B
lstm_22/StatefulPartitionedCalllstm_22/StatefulPartitionedCall2B
lstm_23/StatefulPartitionedCalllstm_23/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ц
G
+__inference_dropout_38_layer_call_fn_685633

inputs
identity┤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_38_layer_call_and_return_conditional_losses_681511`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╦
Ђ
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_680330

inputs

states
states_10
matmul_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         Љ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_namestates:OK
'
_output_shapes
:         
 
_user_specified_namestates
ќ

с
lstm_21_while_cond_682693,
(lstm_21_while_lstm_21_while_loop_counter2
.lstm_21_while_lstm_21_while_maximum_iterations
lstm_21_while_placeholder
lstm_21_while_placeholder_1
lstm_21_while_placeholder_2
lstm_21_while_placeholder_3.
*lstm_21_while_less_lstm_21_strided_slice_1D
@lstm_21_while_lstm_21_while_cond_682693___redundant_placeholder0D
@lstm_21_while_lstm_21_while_cond_682693___redundant_placeholder1D
@lstm_21_while_lstm_21_while_cond_682693___redundant_placeholder2D
@lstm_21_while_lstm_21_while_cond_682693___redundant_placeholder3
lstm_21_while_identity
ѓ
lstm_21/while/LessLesslstm_21_while_placeholder*lstm_21_while_less_lstm_21_strided_slice_1*
T0*
_output_shapes
: [
lstm_21/while/IdentityIdentitylstm_21/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_21_while_identitylstm_21/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
ќ

с
lstm_23_while_cond_682973,
(lstm_23_while_lstm_23_while_loop_counter2
.lstm_23_while_lstm_23_while_maximum_iterations
lstm_23_while_placeholder
lstm_23_while_placeholder_1
lstm_23_while_placeholder_2
lstm_23_while_placeholder_3.
*lstm_23_while_less_lstm_23_strided_slice_1D
@lstm_23_while_lstm_23_while_cond_682973___redundant_placeholder0D
@lstm_23_while_lstm_23_while_cond_682973___redundant_placeholder1D
@lstm_23_while_lstm_23_while_cond_682973___redundant_placeholder2D
@lstm_23_while_lstm_23_while_cond_682973___redundant_placeholder3
lstm_23_while_identity
ѓ
lstm_23/while/LessLesslstm_23_while_placeholder*lstm_23_while_less_lstm_23_strided_slice_1*
T0*
_output_shapes
: [
lstm_23/while/IdentityIdentitylstm_23/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_23_while_identitylstm_23/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
┼
ќ
)__inference_dense_21_layer_call_fn_685617

inputs
unknown:
	unknown_0:
identityѕбStatefulPartitionedCall▄
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_681500o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
њ8
ѓ
C__inference_lstm_21_layer_call_and_return_conditional_losses_680254

inputs%
lstm_cell_21_680172:@%
lstm_cell_21_680174:@!
lstm_cell_21_680176:@
identityѕб$lstm_cell_21/StatefulPartitionedCallбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЭ
$lstm_cell_21/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_21_680172lstm_cell_21_680174lstm_cell_21_680176*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_lstm_cell_21_layer_call_and_return_conditional_losses_680126n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : и
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_21_680172lstm_cell_21_680174lstm_cell_21_680176*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_680185*
condR
while_cond_680184*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  u
NoOpNoOp%^lstm_cell_21/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2L
$lstm_cell_21/StatefulPartitionedCall$lstm_cell_21/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
Ш
d
+__inference_dropout_37_layer_call_fn_685591

inputs
identityѕбStatefulPartitionedCall─
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_681630o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ы
┬
-__inference_sequential_7_layer_call_fn_682408
lstm_21_input
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:
identityѕбStatefulPartitionedCall░
StatefulPartitionedCallStatefulPartitionedCalllstm_21_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *3
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_682332o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:         
'
_user_specified_namelstm_21_input
┼
ќ
)__inference_dense_19_layer_call_fn_685523

inputs
unknown:
	unknown_0:
identityѕбStatefulPartitionedCall▄
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_681452o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
в
з
-__inference_lstm_cell_21_layer_call_fn_685691

inputs
states_0
states_1
unknown:@
	unknown_0:@
	unknown_1:@
identity

identity_1

identity_2ѕбStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_lstm_cell_21_layer_call_and_return_conditional_losses_679980o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
states/0:QM
'
_output_shapes
:         
"
_user_specified_name
states/1
ЎJ
ў
C__inference_lstm_21_layer_call_and_return_conditional_losses_681112

inputs=
+lstm_cell_21_matmul_readvariableop_resource:@?
-lstm_cell_21_matmul_1_readvariableop_resource:@:
,lstm_cell_21_biasadd_readvariableop_resource:@
identityѕб#lstm_cell_21/BiasAdd/ReadVariableOpб"lstm_cell_21/MatMul/ReadVariableOpб$lstm_cell_21/MatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskј
"lstm_cell_21/MatMul/ReadVariableOpReadVariableOp+lstm_cell_21_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ћ
lstm_cell_21/MatMulMatMulstrided_slice_2:output:0*lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @њ
$lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_21_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell_21/MatMul_1MatMulzeros:output:0,lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @І
lstm_cell_21/addAddV2lstm_cell_21/MatMul:product:0lstm_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:         @ї
#lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ћ
lstm_cell_21/BiasAddBiasAddlstm_cell_21/add:z:0+lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @^
lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_21/splitSplit%lstm_cell_21/split/split_dim:output:0lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitn
lstm_cell_21/SigmoidSigmoidlstm_cell_21/split:output:0*
T0*'
_output_shapes
:         p
lstm_cell_21/Sigmoid_1Sigmoidlstm_cell_21/split:output:1*
T0*'
_output_shapes
:         w
lstm_cell_21/mulMullstm_cell_21/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         h
lstm_cell_21/ReluRelulstm_cell_21/split:output:2*
T0*'
_output_shapes
:         є
lstm_cell_21/mul_1Mullstm_cell_21/Sigmoid:y:0lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         {
lstm_cell_21/add_1AddV2lstm_cell_21/mul:z:0lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         p
lstm_cell_21/Sigmoid_2Sigmoidlstm_cell_21/split:output:3*
T0*'
_output_shapes
:         e
lstm_cell_21/Relu_1Relulstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         і
lstm_cell_21/mul_2Mullstm_cell_21/Sigmoid_2:y:0!lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѓ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_21_matmul_readvariableop_resource-lstm_cell_21_matmul_1_readvariableop_resource,lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_681028*
condR
while_cond_681027*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         └
NoOpNoOp$^lstm_cell_21/BiasAdd/ReadVariableOp#^lstm_cell_21/MatMul/ReadVariableOp%^lstm_cell_21/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2J
#lstm_cell_21/BiasAdd/ReadVariableOp#lstm_cell_21/BiasAdd/ReadVariableOp2H
"lstm_cell_21/MatMul/ReadVariableOp"lstm_cell_21/MatMul/ReadVariableOp2L
$lstm_cell_21/MatMul_1/ReadVariableOp$lstm_cell_21/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ОJ
џ
C__inference_lstm_21_layer_call_and_return_conditional_losses_683772
inputs_0=
+lstm_cell_21_matmul_readvariableop_resource:@?
-lstm_cell_21_matmul_1_readvariableop_resource:@:
,lstm_cell_21_biasadd_readvariableop_resource:@
identityѕб#lstm_cell_21/BiasAdd/ReadVariableOpб"lstm_cell_21/MatMul/ReadVariableOpб$lstm_cell_21/MatMul_1/ReadVariableOpбwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskј
"lstm_cell_21/MatMul/ReadVariableOpReadVariableOp+lstm_cell_21_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ћ
lstm_cell_21/MatMulMatMulstrided_slice_2:output:0*lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @њ
$lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_21_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell_21/MatMul_1MatMulzeros:output:0,lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @І
lstm_cell_21/addAddV2lstm_cell_21/MatMul:product:0lstm_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:         @ї
#lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ћ
lstm_cell_21/BiasAddBiasAddlstm_cell_21/add:z:0+lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @^
lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_21/splitSplit%lstm_cell_21/split/split_dim:output:0lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitn
lstm_cell_21/SigmoidSigmoidlstm_cell_21/split:output:0*
T0*'
_output_shapes
:         p
lstm_cell_21/Sigmoid_1Sigmoidlstm_cell_21/split:output:1*
T0*'
_output_shapes
:         w
lstm_cell_21/mulMullstm_cell_21/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         h
lstm_cell_21/ReluRelulstm_cell_21/split:output:2*
T0*'
_output_shapes
:         є
lstm_cell_21/mul_1Mullstm_cell_21/Sigmoid:y:0lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         {
lstm_cell_21/add_1AddV2lstm_cell_21/mul:z:0lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         p
lstm_cell_21/Sigmoid_2Sigmoidlstm_cell_21/split:output:3*
T0*'
_output_shapes
:         e
lstm_cell_21/Relu_1Relulstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         і
lstm_cell_21/mul_2Mullstm_cell_21/Sigmoid_2:y:0!lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѓ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_21_matmul_readvariableop_resource-lstm_cell_21_matmul_1_readvariableop_resource,lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_683688*
condR
while_cond_683687*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  └
NoOpNoOp$^lstm_cell_21/BiasAdd/ReadVariableOp#^lstm_cell_21/MatMul/ReadVariableOp%^lstm_cell_21/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2J
#lstm_cell_21/BiasAdd/ReadVariableOp#lstm_cell_21/BiasAdd/ReadVariableOp2H
"lstm_cell_21/MatMul/ReadVariableOp"lstm_cell_21/MatMul/ReadVariableOp2L
$lstm_cell_21/MatMul_1/ReadVariableOp$lstm_cell_21/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
З	
e
F__inference_dropout_37_layer_call_and_return_conditional_losses_685608

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
с"
П
while_body_680694
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_23_680718_0:@-
while_lstm_cell_23_680720_0:@)
while_lstm_cell_23_680722_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_23_680718:@+
while_lstm_cell_23_680720:@'
while_lstm_cell_23_680722:@ѕб*while/lstm_cell_23/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Х
*while/lstm_cell_23/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_23_680718_0while_lstm_cell_23_680720_0while_lstm_cell_23_680722_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_lstm_cell_23_layer_call_and_return_conditional_losses_680680▄
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_23/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: љ
while/Identity_4Identity3while/lstm_cell_23/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         љ
while/Identity_5Identity3while/lstm_cell_23/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         y

while/NoOpNoOp+^while/lstm_cell_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_23_680718while_lstm_cell_23_680718_0"8
while_lstm_cell_23_680720while_lstm_cell_23_680720_0"8
while_lstm_cell_23_680722while_lstm_cell_23_680722_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2X
*while/lstm_cell_23/StatefulPartitionedCall*while/lstm_cell_23/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
х
├
while_cond_685116
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_685116___redundant_placeholder04
0while_while_cond_685116___redundant_placeholder14
0while_while_cond_685116___redundant_placeholder24
0while_while_cond_685116___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
в
з
-__inference_lstm_cell_22_layer_call_fn_685806

inputs
states_0
states_1
unknown:@
	unknown_0:@
	unknown_1:@
identity

identity_1

identity_2ѕбStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_680476o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
states/0:QM
'
_output_shapes
:         
"
_user_specified_name
states/1
ИA
╩

lstm_23_while_body_683442,
(lstm_23_while_lstm_23_while_loop_counter2
.lstm_23_while_lstm_23_while_maximum_iterations
lstm_23_while_placeholder
lstm_23_while_placeholder_1
lstm_23_while_placeholder_2
lstm_23_while_placeholder_3+
'lstm_23_while_lstm_23_strided_slice_1_0g
clstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensor_0M
;lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0:@O
=lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0:@J
<lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0:@
lstm_23_while_identity
lstm_23_while_identity_1
lstm_23_while_identity_2
lstm_23_while_identity_3
lstm_23_while_identity_4
lstm_23_while_identity_5)
%lstm_23_while_lstm_23_strided_slice_1e
alstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensorK
9lstm_23_while_lstm_cell_23_matmul_readvariableop_resource:@M
;lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource:@H
:lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource:@ѕб1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOpб0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOpб2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOpљ
?lstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╬
1lstm_23/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensor_0lstm_23_while_placeholderHlstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0г
0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp;lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0Л
!lstm_23/while/lstm_cell_23/MatMulMatMul8lstm_23/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @░
2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp=lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0И
#lstm_23/while/lstm_cell_23/MatMul_1MatMullstm_23_while_placeholder_2:lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @х
lstm_23/while/lstm_cell_23/addAddV2+lstm_23/while/lstm_cell_23/MatMul:product:0-lstm_23/while/lstm_cell_23/MatMul_1:product:0*
T0*'
_output_shapes
:         @ф
1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp<lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0Й
"lstm_23/while/lstm_cell_23/BiasAddBiasAdd"lstm_23/while/lstm_cell_23/add:z:09lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @l
*lstm_23/while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Є
 lstm_23/while/lstm_cell_23/splitSplit3lstm_23/while/lstm_cell_23/split/split_dim:output:0+lstm_23/while/lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitі
"lstm_23/while/lstm_cell_23/SigmoidSigmoid)lstm_23/while/lstm_cell_23/split:output:0*
T0*'
_output_shapes
:         ї
$lstm_23/while/lstm_cell_23/Sigmoid_1Sigmoid)lstm_23/while/lstm_cell_23/split:output:1*
T0*'
_output_shapes
:         ъ
lstm_23/while/lstm_cell_23/mulMul(lstm_23/while/lstm_cell_23/Sigmoid_1:y:0lstm_23_while_placeholder_3*
T0*'
_output_shapes
:         ё
lstm_23/while/lstm_cell_23/ReluRelu)lstm_23/while/lstm_cell_23/split:output:2*
T0*'
_output_shapes
:         ░
 lstm_23/while/lstm_cell_23/mul_1Mul&lstm_23/while/lstm_cell_23/Sigmoid:y:0-lstm_23/while/lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         Ц
 lstm_23/while/lstm_cell_23/add_1AddV2"lstm_23/while/lstm_cell_23/mul:z:0$lstm_23/while/lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         ї
$lstm_23/while/lstm_cell_23/Sigmoid_2Sigmoid)lstm_23/while/lstm_cell_23/split:output:3*
T0*'
_output_shapes
:         Ђ
!lstm_23/while/lstm_cell_23/Relu_1Relu$lstm_23/while/lstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         ┤
 lstm_23/while/lstm_cell_23/mul_2Mul(lstm_23/while/lstm_cell_23/Sigmoid_2:y:0/lstm_23/while/lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         т
2lstm_23/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_23_while_placeholder_1lstm_23_while_placeholder$lstm_23/while/lstm_cell_23/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмU
lstm_23/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_23/while/addAddV2lstm_23_while_placeholderlstm_23/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_23/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Є
lstm_23/while/add_1AddV2(lstm_23_while_lstm_23_while_loop_counterlstm_23/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_23/while/IdentityIdentitylstm_23/while/add_1:z:0^lstm_23/while/NoOp*
T0*
_output_shapes
: і
lstm_23/while/Identity_1Identity.lstm_23_while_lstm_23_while_maximum_iterations^lstm_23/while/NoOp*
T0*
_output_shapes
: q
lstm_23/while/Identity_2Identitylstm_23/while/add:z:0^lstm_23/while/NoOp*
T0*
_output_shapes
: ъ
lstm_23/while/Identity_3IdentityBlstm_23/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_23/while/NoOp*
T0*
_output_shapes
: Љ
lstm_23/while/Identity_4Identity$lstm_23/while/lstm_cell_23/mul_2:z:0^lstm_23/while/NoOp*
T0*'
_output_shapes
:         Љ
lstm_23/while/Identity_5Identity$lstm_23/while/lstm_cell_23/add_1:z:0^lstm_23/while/NoOp*
T0*'
_output_shapes
:         ­
lstm_23/while/NoOpNoOp2^lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp1^lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp3^lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_23_while_identitylstm_23/while/Identity:output:0"=
lstm_23_while_identity_1!lstm_23/while/Identity_1:output:0"=
lstm_23_while_identity_2!lstm_23/while/Identity_2:output:0"=
lstm_23_while_identity_3!lstm_23/while/Identity_3:output:0"=
lstm_23_while_identity_4!lstm_23/while/Identity_4:output:0"=
lstm_23_while_identity_5!lstm_23/while/Identity_5:output:0"P
%lstm_23_while_lstm_23_strided_slice_1'lstm_23_while_lstm_23_strided_slice_1_0"z
:lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource<lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0"|
;lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource=lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0"x
9lstm_23_while_lstm_cell_23_matmul_readvariableop_resource;lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0"╚
alstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensorclstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2f
1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp2d
0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp2h
2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
ъJ
ў
C__inference_lstm_23_layer_call_and_return_conditional_losses_685344

inputs=
+lstm_cell_23_matmul_readvariableop_resource:@?
-lstm_cell_23_matmul_1_readvariableop_resource:@:
,lstm_cell_23_biasadd_readvariableop_resource:@
identityѕб#lstm_cell_23/BiasAdd/ReadVariableOpб"lstm_cell_23/MatMul/ReadVariableOpб$lstm_cell_23/MatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskј
"lstm_cell_23/MatMul/ReadVariableOpReadVariableOp+lstm_cell_23_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ћ
lstm_cell_23/MatMulMatMulstrided_slice_2:output:0*lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @њ
$lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_23_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell_23/MatMul_1MatMulzeros:output:0,lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @І
lstm_cell_23/addAddV2lstm_cell_23/MatMul:product:0lstm_cell_23/MatMul_1:product:0*
T0*'
_output_shapes
:         @ї
#lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ћ
lstm_cell_23/BiasAddBiasAddlstm_cell_23/add:z:0+lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @^
lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_23/splitSplit%lstm_cell_23/split/split_dim:output:0lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitn
lstm_cell_23/SigmoidSigmoidlstm_cell_23/split:output:0*
T0*'
_output_shapes
:         p
lstm_cell_23/Sigmoid_1Sigmoidlstm_cell_23/split:output:1*
T0*'
_output_shapes
:         w
lstm_cell_23/mulMullstm_cell_23/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         h
lstm_cell_23/ReluRelulstm_cell_23/split:output:2*
T0*'
_output_shapes
:         є
lstm_cell_23/mul_1Mullstm_cell_23/Sigmoid:y:0lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         {
lstm_cell_23/add_1AddV2lstm_cell_23/mul:z:0lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         p
lstm_cell_23/Sigmoid_2Sigmoidlstm_cell_23/split:output:3*
T0*'
_output_shapes
:         e
lstm_cell_23/Relu_1Relulstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         і
lstm_cell_23/mul_2Mullstm_cell_23/Sigmoid_2:y:0!lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѓ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_23_matmul_readvariableop_resource-lstm_cell_23_matmul_1_readvariableop_resource,lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_685260*
condR
while_cond_685259*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         └
NoOpNoOp$^lstm_cell_23/BiasAdd/ReadVariableOp#^lstm_cell_23/MatMul/ReadVariableOp%^lstm_cell_23/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2J
#lstm_cell_23/BiasAdd/ReadVariableOp#lstm_cell_23/BiasAdd/ReadVariableOp2H
"lstm_cell_23/MatMul/ReadVariableOp"lstm_cell_23/MatMul/ReadVariableOp2L
$lstm_cell_23/MatMul_1/ReadVariableOp$lstm_cell_23/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
в
з
-__inference_lstm_cell_23_layer_call_fn_685904

inputs
states_0
states_1
unknown:@
	unknown_0:@
	unknown_1:@
identity

identity_1

identity_2ѕбStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_lstm_cell_23_layer_call_and_return_conditional_losses_680826o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
states/0:QM
'
_output_shapes
:         
"
_user_specified_name
states/1
Џ

ш
D__inference_dense_19_layer_call_and_return_conditional_losses_685534

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ЎJ
ў
C__inference_lstm_21_layer_call_and_return_conditional_losses_684201

inputs=
+lstm_cell_21_matmul_readvariableop_resource:@?
-lstm_cell_21_matmul_1_readvariableop_resource:@:
,lstm_cell_21_biasadd_readvariableop_resource:@
identityѕб#lstm_cell_21/BiasAdd/ReadVariableOpб"lstm_cell_21/MatMul/ReadVariableOpб$lstm_cell_21/MatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskј
"lstm_cell_21/MatMul/ReadVariableOpReadVariableOp+lstm_cell_21_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ћ
lstm_cell_21/MatMulMatMulstrided_slice_2:output:0*lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @њ
$lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_21_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell_21/MatMul_1MatMulzeros:output:0,lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @І
lstm_cell_21/addAddV2lstm_cell_21/MatMul:product:0lstm_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:         @ї
#lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ћ
lstm_cell_21/BiasAddBiasAddlstm_cell_21/add:z:0+lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @^
lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_21/splitSplit%lstm_cell_21/split/split_dim:output:0lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitn
lstm_cell_21/SigmoidSigmoidlstm_cell_21/split:output:0*
T0*'
_output_shapes
:         p
lstm_cell_21/Sigmoid_1Sigmoidlstm_cell_21/split:output:1*
T0*'
_output_shapes
:         w
lstm_cell_21/mulMullstm_cell_21/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         h
lstm_cell_21/ReluRelulstm_cell_21/split:output:2*
T0*'
_output_shapes
:         є
lstm_cell_21/mul_1Mullstm_cell_21/Sigmoid:y:0lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         {
lstm_cell_21/add_1AddV2lstm_cell_21/mul:z:0lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         p
lstm_cell_21/Sigmoid_2Sigmoidlstm_cell_21/split:output:3*
T0*'
_output_shapes
:         e
lstm_cell_21/Relu_1Relulstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         і
lstm_cell_21/mul_2Mullstm_cell_21/Sigmoid_2:y:0!lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѓ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_21_matmul_readvariableop_resource-lstm_cell_21_matmul_1_readvariableop_resource,lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_684117*
condR
while_cond_684116*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         └
NoOpNoOp$^lstm_cell_21/BiasAdd/ReadVariableOp#^lstm_cell_21/MatMul/ReadVariableOp%^lstm_cell_21/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2J
#lstm_cell_21/BiasAdd/ReadVariableOp#lstm_cell_21/BiasAdd/ReadVariableOp2H
"lstm_cell_21/MatMul/ReadVariableOp"lstm_cell_21/MatMul/ReadVariableOp2L
$lstm_cell_21/MatMul_1/ReadVariableOp$lstm_cell_21/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Е
┤
(__inference_lstm_22_layer_call_fn_684250
inputs_0
unknown:@
	unknown_0:@
	unknown_1:@
identityѕбStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_680604|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
х
├
while_cond_684330
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_684330___redundant_placeholder04
0while_while_cond_684330___redundant_placeholder14
0while_while_cond_684330___redundant_placeholder24
0while_while_cond_684330___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
х
├
while_cond_681184
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_681184___redundant_placeholder04
0while_while_cond_681184___redundant_placeholder14
0while_while_cond_681184___redundant_placeholder24
0while_while_cond_681184___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
х
├
while_cond_683830
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_683830___redundant_placeholder04
0while_while_cond_683830___redundant_placeholder14
0while_while_cond_683830___redundant_placeholder24
0while_while_cond_683830___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
ОJ
џ
C__inference_lstm_22_layer_call_and_return_conditional_losses_684415
inputs_0=
+lstm_cell_22_matmul_readvariableop_resource:@?
-lstm_cell_22_matmul_1_readvariableop_resource:@:
,lstm_cell_22_biasadd_readvariableop_resource:@
identityѕб#lstm_cell_22/BiasAdd/ReadVariableOpб"lstm_cell_22/MatMul/ReadVariableOpб$lstm_cell_22/MatMul_1/ReadVariableOpбwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskј
"lstm_cell_22/MatMul/ReadVariableOpReadVariableOp+lstm_cell_22_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ћ
lstm_cell_22/MatMulMatMulstrided_slice_2:output:0*lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @њ
$lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_22_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell_22/MatMul_1MatMulzeros:output:0,lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @І
lstm_cell_22/addAddV2lstm_cell_22/MatMul:product:0lstm_cell_22/MatMul_1:product:0*
T0*'
_output_shapes
:         @ї
#lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ћ
lstm_cell_22/BiasAddBiasAddlstm_cell_22/add:z:0+lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @^
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0lstm_cell_22/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitn
lstm_cell_22/SigmoidSigmoidlstm_cell_22/split:output:0*
T0*'
_output_shapes
:         p
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/split:output:1*
T0*'
_output_shapes
:         w
lstm_cell_22/mulMullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         h
lstm_cell_22/ReluRelulstm_cell_22/split:output:2*
T0*'
_output_shapes
:         є
lstm_cell_22/mul_1Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Relu:activations:0*
T0*'
_output_shapes
:         {
lstm_cell_22/add_1AddV2lstm_cell_22/mul:z:0lstm_cell_22/mul_1:z:0*
T0*'
_output_shapes
:         p
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/split:output:3*
T0*'
_output_shapes
:         e
lstm_cell_22/Relu_1Relulstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         і
lstm_cell_22/mul_2Mullstm_cell_22/Sigmoid_2:y:0!lstm_cell_22/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѓ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_22_matmul_readvariableop_resource-lstm_cell_22_matmul_1_readvariableop_resource,lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_684331*
condR
while_cond_684330*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  └
NoOpNoOp$^lstm_cell_22/BiasAdd/ReadVariableOp#^lstm_cell_22/MatMul/ReadVariableOp%^lstm_cell_22/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2J
#lstm_cell_22/BiasAdd/ReadVariableOp#lstm_cell_22/BiasAdd/ReadVariableOp2H
"lstm_cell_22/MatMul/ReadVariableOp"lstm_cell_22/MatMul/ReadVariableOp2L
$lstm_cell_22/MatMul_1/ReadVariableOp$lstm_cell_22/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
ЯP
в
&sequential_7_lstm_22_while_body_679658F
Bsequential_7_lstm_22_while_sequential_7_lstm_22_while_loop_counterL
Hsequential_7_lstm_22_while_sequential_7_lstm_22_while_maximum_iterations*
&sequential_7_lstm_22_while_placeholder,
(sequential_7_lstm_22_while_placeholder_1,
(sequential_7_lstm_22_while_placeholder_2,
(sequential_7_lstm_22_while_placeholder_3E
Asequential_7_lstm_22_while_sequential_7_lstm_22_strided_slice_1_0Ђ
}sequential_7_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_22_tensorarrayunstack_tensorlistfromtensor_0Z
Hsequential_7_lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0:@\
Jsequential_7_lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0:@W
Isequential_7_lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0:@'
#sequential_7_lstm_22_while_identity)
%sequential_7_lstm_22_while_identity_1)
%sequential_7_lstm_22_while_identity_2)
%sequential_7_lstm_22_while_identity_3)
%sequential_7_lstm_22_while_identity_4)
%sequential_7_lstm_22_while_identity_5C
?sequential_7_lstm_22_while_sequential_7_lstm_22_strided_slice_1
{sequential_7_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_22_tensorarrayunstack_tensorlistfromtensorX
Fsequential_7_lstm_22_while_lstm_cell_22_matmul_readvariableop_resource:@Z
Hsequential_7_lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource:@U
Gsequential_7_lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource:@ѕб>sequential_7/lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOpб=sequential_7/lstm_22/while/lstm_cell_22/MatMul/ReadVariableOpб?sequential_7/lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpЮ
Lsequential_7/lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Ј
>sequential_7/lstm_22/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_7_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_22_tensorarrayunstack_tensorlistfromtensor_0&sequential_7_lstm_22_while_placeholderUsequential_7/lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0к
=sequential_7/lstm_22/while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOpHsequential_7_lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0Э
.sequential_7/lstm_22/while/lstm_cell_22/MatMulMatMulEsequential_7/lstm_22/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_7/lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @╩
?sequential_7/lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOpJsequential_7_lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0▀
0sequential_7/lstm_22/while/lstm_cell_22/MatMul_1MatMul(sequential_7_lstm_22_while_placeholder_2Gsequential_7/lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @▄
+sequential_7/lstm_22/while/lstm_cell_22/addAddV28sequential_7/lstm_22/while/lstm_cell_22/MatMul:product:0:sequential_7/lstm_22/while/lstm_cell_22/MatMul_1:product:0*
T0*'
_output_shapes
:         @─
>sequential_7/lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOpIsequential_7_lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0т
/sequential_7/lstm_22/while/lstm_cell_22/BiasAddBiasAdd/sequential_7/lstm_22/while/lstm_cell_22/add:z:0Fsequential_7/lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @y
7sequential_7/lstm_22/while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :«
-sequential_7/lstm_22/while/lstm_cell_22/splitSplit@sequential_7/lstm_22/while/lstm_cell_22/split/split_dim:output:08sequential_7/lstm_22/while/lstm_cell_22/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitц
/sequential_7/lstm_22/while/lstm_cell_22/SigmoidSigmoid6sequential_7/lstm_22/while/lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         д
1sequential_7/lstm_22/while/lstm_cell_22/Sigmoid_1Sigmoid6sequential_7/lstm_22/while/lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         ┼
+sequential_7/lstm_22/while/lstm_cell_22/mulMul5sequential_7/lstm_22/while/lstm_cell_22/Sigmoid_1:y:0(sequential_7_lstm_22_while_placeholder_3*
T0*'
_output_shapes
:         ъ
,sequential_7/lstm_22/while/lstm_cell_22/ReluRelu6sequential_7/lstm_22/while/lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         О
-sequential_7/lstm_22/while/lstm_cell_22/mul_1Mul3sequential_7/lstm_22/while/lstm_cell_22/Sigmoid:y:0:sequential_7/lstm_22/while/lstm_cell_22/Relu:activations:0*
T0*'
_output_shapes
:         ╠
-sequential_7/lstm_22/while/lstm_cell_22/add_1AddV2/sequential_7/lstm_22/while/lstm_cell_22/mul:z:01sequential_7/lstm_22/while/lstm_cell_22/mul_1:z:0*
T0*'
_output_shapes
:         д
1sequential_7/lstm_22/while/lstm_cell_22/Sigmoid_2Sigmoid6sequential_7/lstm_22/while/lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         Џ
.sequential_7/lstm_22/while/lstm_cell_22/Relu_1Relu1sequential_7/lstm_22/while/lstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         █
-sequential_7/lstm_22/while/lstm_cell_22/mul_2Mul5sequential_7/lstm_22/while/lstm_cell_22/Sigmoid_2:y:0<sequential_7/lstm_22/while/lstm_cell_22/Relu_1:activations:0*
T0*'
_output_shapes
:         Ў
?sequential_7/lstm_22/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_7_lstm_22_while_placeholder_1&sequential_7_lstm_22_while_placeholder1sequential_7/lstm_22/while/lstm_cell_22/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмb
 sequential_7/lstm_22/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Џ
sequential_7/lstm_22/while/addAddV2&sequential_7_lstm_22_while_placeholder)sequential_7/lstm_22/while/add/y:output:0*
T0*
_output_shapes
: d
"sequential_7/lstm_22/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :╗
 sequential_7/lstm_22/while/add_1AddV2Bsequential_7_lstm_22_while_sequential_7_lstm_22_while_loop_counter+sequential_7/lstm_22/while/add_1/y:output:0*
T0*
_output_shapes
: ў
#sequential_7/lstm_22/while/IdentityIdentity$sequential_7/lstm_22/while/add_1:z:0 ^sequential_7/lstm_22/while/NoOp*
T0*
_output_shapes
: Й
%sequential_7/lstm_22/while/Identity_1IdentityHsequential_7_lstm_22_while_sequential_7_lstm_22_while_maximum_iterations ^sequential_7/lstm_22/while/NoOp*
T0*
_output_shapes
: ў
%sequential_7/lstm_22/while/Identity_2Identity"sequential_7/lstm_22/while/add:z:0 ^sequential_7/lstm_22/while/NoOp*
T0*
_output_shapes
: ┼
%sequential_7/lstm_22/while/Identity_3IdentityOsequential_7/lstm_22/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_7/lstm_22/while/NoOp*
T0*
_output_shapes
: И
%sequential_7/lstm_22/while/Identity_4Identity1sequential_7/lstm_22/while/lstm_cell_22/mul_2:z:0 ^sequential_7/lstm_22/while/NoOp*
T0*'
_output_shapes
:         И
%sequential_7/lstm_22/while/Identity_5Identity1sequential_7/lstm_22/while/lstm_cell_22/add_1:z:0 ^sequential_7/lstm_22/while/NoOp*
T0*'
_output_shapes
:         ц
sequential_7/lstm_22/while/NoOpNoOp?^sequential_7/lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp>^sequential_7/lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp@^sequential_7/lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "S
#sequential_7_lstm_22_while_identity,sequential_7/lstm_22/while/Identity:output:0"W
%sequential_7_lstm_22_while_identity_1.sequential_7/lstm_22/while/Identity_1:output:0"W
%sequential_7_lstm_22_while_identity_2.sequential_7/lstm_22/while/Identity_2:output:0"W
%sequential_7_lstm_22_while_identity_3.sequential_7/lstm_22/while/Identity_3:output:0"W
%sequential_7_lstm_22_while_identity_4.sequential_7/lstm_22/while/Identity_4:output:0"W
%sequential_7_lstm_22_while_identity_5.sequential_7/lstm_22/while/Identity_5:output:0"ћ
Gsequential_7_lstm_22_while_lstm_cell_22_biasadd_readvariableop_resourceIsequential_7_lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0"ќ
Hsequential_7_lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resourceJsequential_7_lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0"њ
Fsequential_7_lstm_22_while_lstm_cell_22_matmul_readvariableop_resourceHsequential_7_lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0"ё
?sequential_7_lstm_22_while_sequential_7_lstm_22_strided_slice_1Asequential_7_lstm_22_while_sequential_7_lstm_22_strided_slice_1_0"Ч
{sequential_7_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_22_tensorarrayunstack_tensorlistfromtensor}sequential_7_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_22_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2ђ
>sequential_7/lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp>sequential_7/lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp2~
=sequential_7/lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp=sequential_7/lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp2ѓ
?sequential_7/lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp?sequential_7/lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
ЎJ
ў
C__inference_lstm_22_layer_call_and_return_conditional_losses_682043

inputs=
+lstm_cell_22_matmul_readvariableop_resource:@?
-lstm_cell_22_matmul_1_readvariableop_resource:@:
,lstm_cell_22_biasadd_readvariableop_resource:@
identityѕб#lstm_cell_22/BiasAdd/ReadVariableOpб"lstm_cell_22/MatMul/ReadVariableOpб$lstm_cell_22/MatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskј
"lstm_cell_22/MatMul/ReadVariableOpReadVariableOp+lstm_cell_22_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ћ
lstm_cell_22/MatMulMatMulstrided_slice_2:output:0*lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @њ
$lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_22_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell_22/MatMul_1MatMulzeros:output:0,lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @І
lstm_cell_22/addAddV2lstm_cell_22/MatMul:product:0lstm_cell_22/MatMul_1:product:0*
T0*'
_output_shapes
:         @ї
#lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ћ
lstm_cell_22/BiasAddBiasAddlstm_cell_22/add:z:0+lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @^
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0lstm_cell_22/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitn
lstm_cell_22/SigmoidSigmoidlstm_cell_22/split:output:0*
T0*'
_output_shapes
:         p
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/split:output:1*
T0*'
_output_shapes
:         w
lstm_cell_22/mulMullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         h
lstm_cell_22/ReluRelulstm_cell_22/split:output:2*
T0*'
_output_shapes
:         є
lstm_cell_22/mul_1Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Relu:activations:0*
T0*'
_output_shapes
:         {
lstm_cell_22/add_1AddV2lstm_cell_22/mul:z:0lstm_cell_22/mul_1:z:0*
T0*'
_output_shapes
:         p
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/split:output:3*
T0*'
_output_shapes
:         e
lstm_cell_22/Relu_1Relulstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         і
lstm_cell_22/mul_2Mullstm_cell_22/Sigmoid_2:y:0!lstm_cell_22/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѓ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_22_matmul_readvariableop_resource-lstm_cell_22_matmul_1_readvariableop_resource,lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_681959*
condR
while_cond_681958*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         └
NoOpNoOp$^lstm_cell_22/BiasAdd/ReadVariableOp#^lstm_cell_22/MatMul/ReadVariableOp%^lstm_cell_22/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2J
#lstm_cell_22/BiasAdd/ReadVariableOp#lstm_cell_22/BiasAdd/ReadVariableOp2H
"lstm_cell_22/MatMul/ReadVariableOp"lstm_cell_22/MatMul/ReadVariableOp2L
$lstm_cell_22/MatMul_1/ReadVariableOp$lstm_cell_22/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
х
├
while_cond_681770
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_681770___redundant_placeholder04
0while_while_cond_681770___redundant_placeholder14
0while_while_cond_681770___redundant_placeholder24
0while_while_cond_681770___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
З	
e
F__inference_dropout_38_layer_call_and_return_conditional_losses_681597

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Нџ
Ѕ
H__inference_sequential_7_layer_call_and_return_conditional_losses_683089

inputsE
3lstm_21_lstm_cell_21_matmul_readvariableop_resource:@G
5lstm_21_lstm_cell_21_matmul_1_readvariableop_resource:@B
4lstm_21_lstm_cell_21_biasadd_readvariableop_resource:@E
3lstm_22_lstm_cell_22_matmul_readvariableop_resource:@G
5lstm_22_lstm_cell_22_matmul_1_readvariableop_resource:@B
4lstm_22_lstm_cell_22_biasadd_readvariableop_resource:@E
3lstm_23_lstm_cell_23_matmul_readvariableop_resource:@G
5lstm_23_lstm_cell_23_matmul_1_readvariableop_resource:@B
4lstm_23_lstm_cell_23_biasadd_readvariableop_resource:@9
'dense_19_matmul_readvariableop_resource:6
(dense_19_biasadd_readvariableop_resource:9
'dense_20_matmul_readvariableop_resource:6
(dense_20_biasadd_readvariableop_resource:9
'dense_21_matmul_readvariableop_resource:6
(dense_21_biasadd_readvariableop_resource:9
'dense_22_matmul_readvariableop_resource:6
(dense_22_biasadd_readvariableop_resource:
identityѕбdense_19/BiasAdd/ReadVariableOpбdense_19/MatMul/ReadVariableOpбdense_20/BiasAdd/ReadVariableOpбdense_20/MatMul/ReadVariableOpбdense_21/BiasAdd/ReadVariableOpбdense_21/MatMul/ReadVariableOpбdense_22/BiasAdd/ReadVariableOpбdense_22/MatMul/ReadVariableOpб+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOpб*lstm_21/lstm_cell_21/MatMul/ReadVariableOpб,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOpбlstm_21/whileб+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOpб*lstm_22/lstm_cell_22/MatMul/ReadVariableOpб,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOpбlstm_22/whileб+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOpб*lstm_23/lstm_cell_23/MatMul/ReadVariableOpб,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOpбlstm_23/whileC
lstm_21/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
lstm_21/strided_sliceStridedSlicelstm_21/Shape:output:0$lstm_21/strided_slice/stack:output:0&lstm_21/strided_slice/stack_1:output:0&lstm_21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_21/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :І
lstm_21/zeros/packedPacklstm_21/strided_slice:output:0lstm_21/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_21/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ё
lstm_21/zerosFilllstm_21/zeros/packed:output:0lstm_21/zeros/Const:output:0*
T0*'
_output_shapes
:         Z
lstm_21/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ј
lstm_21/zeros_1/packedPacklstm_21/strided_slice:output:0!lstm_21/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_21/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    і
lstm_21/zeros_1Filllstm_21/zeros_1/packed:output:0lstm_21/zeros_1/Const:output:0*
T0*'
_output_shapes
:         k
lstm_21/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_21/transpose	Transposeinputslstm_21/transpose/perm:output:0*
T0*+
_output_shapes
:         T
lstm_21/Shape_1Shapelstm_21/transpose:y:0*
T0*
_output_shapes
:g
lstm_21/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_21/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_21/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
lstm_21/strided_slice_1StridedSlicelstm_21/Shape_1:output:0&lstm_21/strided_slice_1/stack:output:0(lstm_21/strided_slice_1/stack_1:output:0(lstm_21/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_21/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╠
lstm_21/TensorArrayV2TensorListReserve,lstm_21/TensorArrayV2/element_shape:output:0 lstm_21/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмј
=lstm_21/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Э
/lstm_21/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_21/transpose:y:0Flstm_21/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмg
lstm_21/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_21/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_21/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Љ
lstm_21/strided_slice_2StridedSlicelstm_21/transpose:y:0&lstm_21/strided_slice_2/stack:output:0(lstm_21/strided_slice_2/stack_1:output:0(lstm_21/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskъ
*lstm_21/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3lstm_21_lstm_cell_21_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Г
lstm_21/lstm_cell_21/MatMulMatMul lstm_21/strided_slice_2:output:02lstm_21/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @б
,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5lstm_21_lstm_cell_21_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Д
lstm_21/lstm_cell_21/MatMul_1MatMullstm_21/zeros:output:04lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Б
lstm_21/lstm_cell_21/addAddV2%lstm_21/lstm_cell_21/MatMul:product:0'lstm_21/lstm_cell_21/MatMul_1:product:0*
T0*'
_output_shapes
:         @ю
+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4lstm_21_lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0г
lstm_21/lstm_cell_21/BiasAddBiasAddlstm_21/lstm_cell_21/add:z:03lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @f
$lstm_21/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ш
lstm_21/lstm_cell_21/splitSplit-lstm_21/lstm_cell_21/split/split_dim:output:0%lstm_21/lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_split~
lstm_21/lstm_cell_21/SigmoidSigmoid#lstm_21/lstm_cell_21/split:output:0*
T0*'
_output_shapes
:         ђ
lstm_21/lstm_cell_21/Sigmoid_1Sigmoid#lstm_21/lstm_cell_21/split:output:1*
T0*'
_output_shapes
:         Ј
lstm_21/lstm_cell_21/mulMul"lstm_21/lstm_cell_21/Sigmoid_1:y:0lstm_21/zeros_1:output:0*
T0*'
_output_shapes
:         x
lstm_21/lstm_cell_21/ReluRelu#lstm_21/lstm_cell_21/split:output:2*
T0*'
_output_shapes
:         ъ
lstm_21/lstm_cell_21/mul_1Mul lstm_21/lstm_cell_21/Sigmoid:y:0'lstm_21/lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         Њ
lstm_21/lstm_cell_21/add_1AddV2lstm_21/lstm_cell_21/mul:z:0lstm_21/lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         ђ
lstm_21/lstm_cell_21/Sigmoid_2Sigmoid#lstm_21/lstm_cell_21/split:output:3*
T0*'
_output_shapes
:         u
lstm_21/lstm_cell_21/Relu_1Relulstm_21/lstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         б
lstm_21/lstm_cell_21/mul_2Mul"lstm_21/lstm_cell_21/Sigmoid_2:y:0)lstm_21/lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         v
%lstm_21/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       л
lstm_21/TensorArrayV2_1TensorListReserve.lstm_21/TensorArrayV2_1/element_shape:output:0 lstm_21/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмN
lstm_21/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_21/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         \
lstm_21/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ы
lstm_21/whileWhile#lstm_21/while/loop_counter:output:0)lstm_21/while/maximum_iterations:output:0lstm_21/time:output:0 lstm_21/TensorArrayV2_1:handle:0lstm_21/zeros:output:0lstm_21/zeros_1:output:0 lstm_21/strided_slice_1:output:0?lstm_21/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_21_lstm_cell_21_matmul_readvariableop_resource5lstm_21_lstm_cell_21_matmul_1_readvariableop_resource4lstm_21_lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_21_while_body_682694*%
condR
lstm_21_while_cond_682693*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ѕ
8lstm_21/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┌
*lstm_21/TensorArrayV2Stack/TensorListStackTensorListStacklstm_21/while:output:3Alstm_21/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0p
lstm_21/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         i
lstm_21/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_21/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:»
lstm_21/strided_slice_3StridedSlice3lstm_21/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_21/strided_slice_3/stack:output:0(lstm_21/strided_slice_3/stack_1:output:0(lstm_21/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskm
lstm_21/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
lstm_21/transpose_1	Transpose3lstm_21/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_21/transpose_1/perm:output:0*
T0*+
_output_shapes
:         c
lstm_21/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    n
dropout_33/IdentityIdentitylstm_21/transpose_1:y:0*
T0*+
_output_shapes
:         Y
lstm_22/ShapeShapedropout_33/Identity:output:0*
T0*
_output_shapes
:e
lstm_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
lstm_22/strided_sliceStridedSlicelstm_22/Shape:output:0$lstm_22/strided_slice/stack:output:0&lstm_22/strided_slice/stack_1:output:0&lstm_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_22/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :І
lstm_22/zeros/packedPacklstm_22/strided_slice:output:0lstm_22/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_22/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ё
lstm_22/zerosFilllstm_22/zeros/packed:output:0lstm_22/zeros/Const:output:0*
T0*'
_output_shapes
:         Z
lstm_22/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ј
lstm_22/zeros_1/packedPacklstm_22/strided_slice:output:0!lstm_22/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_22/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    і
lstm_22/zeros_1Filllstm_22/zeros_1/packed:output:0lstm_22/zeros_1/Const:output:0*
T0*'
_output_shapes
:         k
lstm_22/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Њ
lstm_22/transpose	Transposedropout_33/Identity:output:0lstm_22/transpose/perm:output:0*
T0*+
_output_shapes
:         T
lstm_22/Shape_1Shapelstm_22/transpose:y:0*
T0*
_output_shapes
:g
lstm_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
lstm_22/strided_slice_1StridedSlicelstm_22/Shape_1:output:0&lstm_22/strided_slice_1/stack:output:0(lstm_22/strided_slice_1/stack_1:output:0(lstm_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_22/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╠
lstm_22/TensorArrayV2TensorListReserve,lstm_22/TensorArrayV2/element_shape:output:0 lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмј
=lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Э
/lstm_22/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_22/transpose:y:0Flstm_22/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмg
lstm_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Љ
lstm_22/strided_slice_2StridedSlicelstm_22/transpose:y:0&lstm_22/strided_slice_2/stack:output:0(lstm_22/strided_slice_2/stack_1:output:0(lstm_22/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskъ
*lstm_22/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3lstm_22_lstm_cell_22_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Г
lstm_22/lstm_cell_22/MatMulMatMul lstm_22/strided_slice_2:output:02lstm_22/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @б
,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5lstm_22_lstm_cell_22_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Д
lstm_22/lstm_cell_22/MatMul_1MatMullstm_22/zeros:output:04lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Б
lstm_22/lstm_cell_22/addAddV2%lstm_22/lstm_cell_22/MatMul:product:0'lstm_22/lstm_cell_22/MatMul_1:product:0*
T0*'
_output_shapes
:         @ю
+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4lstm_22_lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0г
lstm_22/lstm_cell_22/BiasAddBiasAddlstm_22/lstm_cell_22/add:z:03lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @f
$lstm_22/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ш
lstm_22/lstm_cell_22/splitSplit-lstm_22/lstm_cell_22/split/split_dim:output:0%lstm_22/lstm_cell_22/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_split~
lstm_22/lstm_cell_22/SigmoidSigmoid#lstm_22/lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         ђ
lstm_22/lstm_cell_22/Sigmoid_1Sigmoid#lstm_22/lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         Ј
lstm_22/lstm_cell_22/mulMul"lstm_22/lstm_cell_22/Sigmoid_1:y:0lstm_22/zeros_1:output:0*
T0*'
_output_shapes
:         x
lstm_22/lstm_cell_22/ReluRelu#lstm_22/lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         ъ
lstm_22/lstm_cell_22/mul_1Mul lstm_22/lstm_cell_22/Sigmoid:y:0'lstm_22/lstm_cell_22/Relu:activations:0*
T0*'
_output_shapes
:         Њ
lstm_22/lstm_cell_22/add_1AddV2lstm_22/lstm_cell_22/mul:z:0lstm_22/lstm_cell_22/mul_1:z:0*
T0*'
_output_shapes
:         ђ
lstm_22/lstm_cell_22/Sigmoid_2Sigmoid#lstm_22/lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         u
lstm_22/lstm_cell_22/Relu_1Relulstm_22/lstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         б
lstm_22/lstm_cell_22/mul_2Mul"lstm_22/lstm_cell_22/Sigmoid_2:y:0)lstm_22/lstm_cell_22/Relu_1:activations:0*
T0*'
_output_shapes
:         v
%lstm_22/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       л
lstm_22/TensorArrayV2_1TensorListReserve.lstm_22/TensorArrayV2_1/element_shape:output:0 lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмN
lstm_22/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_22/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         \
lstm_22/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ы
lstm_22/whileWhile#lstm_22/while/loop_counter:output:0)lstm_22/while/maximum_iterations:output:0lstm_22/time:output:0 lstm_22/TensorArrayV2_1:handle:0lstm_22/zeros:output:0lstm_22/zeros_1:output:0 lstm_22/strided_slice_1:output:0?lstm_22/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_22_lstm_cell_22_matmul_readvariableop_resource5lstm_22_lstm_cell_22_matmul_1_readvariableop_resource4lstm_22_lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_22_while_body_682834*%
condR
lstm_22_while_cond_682833*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ѕ
8lstm_22/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┌
*lstm_22/TensorArrayV2Stack/TensorListStackTensorListStacklstm_22/while:output:3Alstm_22/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0p
lstm_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         i
lstm_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:»
lstm_22/strided_slice_3StridedSlice3lstm_22/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_22/strided_slice_3/stack:output:0(lstm_22/strided_slice_3/stack_1:output:0(lstm_22/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskm
lstm_22/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
lstm_22/transpose_1	Transpose3lstm_22/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_22/transpose_1/perm:output:0*
T0*+
_output_shapes
:         c
lstm_22/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    n
dropout_34/IdentityIdentitylstm_22/transpose_1:y:0*
T0*+
_output_shapes
:         Y
lstm_23/ShapeShapedropout_34/Identity:output:0*
T0*
_output_shapes
:e
lstm_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
lstm_23/strided_sliceStridedSlicelstm_23/Shape:output:0$lstm_23/strided_slice/stack:output:0&lstm_23/strided_slice/stack_1:output:0&lstm_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_23/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :І
lstm_23/zeros/packedPacklstm_23/strided_slice:output:0lstm_23/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_23/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ё
lstm_23/zerosFilllstm_23/zeros/packed:output:0lstm_23/zeros/Const:output:0*
T0*'
_output_shapes
:         Z
lstm_23/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ј
lstm_23/zeros_1/packedPacklstm_23/strided_slice:output:0!lstm_23/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_23/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    і
lstm_23/zeros_1Filllstm_23/zeros_1/packed:output:0lstm_23/zeros_1/Const:output:0*
T0*'
_output_shapes
:         k
lstm_23/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Њ
lstm_23/transpose	Transposedropout_34/Identity:output:0lstm_23/transpose/perm:output:0*
T0*+
_output_shapes
:         T
lstm_23/Shape_1Shapelstm_23/transpose:y:0*
T0*
_output_shapes
:g
lstm_23/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_23/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_23/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
lstm_23/strided_slice_1StridedSlicelstm_23/Shape_1:output:0&lstm_23/strided_slice_1/stack:output:0(lstm_23/strided_slice_1/stack_1:output:0(lstm_23/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_23/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╠
lstm_23/TensorArrayV2TensorListReserve,lstm_23/TensorArrayV2/element_shape:output:0 lstm_23/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмј
=lstm_23/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Э
/lstm_23/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_23/transpose:y:0Flstm_23/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмg
lstm_23/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_23/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_23/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Љ
lstm_23/strided_slice_2StridedSlicelstm_23/transpose:y:0&lstm_23/strided_slice_2/stack:output:0(lstm_23/strided_slice_2/stack_1:output:0(lstm_23/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskъ
*lstm_23/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3lstm_23_lstm_cell_23_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Г
lstm_23/lstm_cell_23/MatMulMatMul lstm_23/strided_slice_2:output:02lstm_23/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @б
,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5lstm_23_lstm_cell_23_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Д
lstm_23/lstm_cell_23/MatMul_1MatMullstm_23/zeros:output:04lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Б
lstm_23/lstm_cell_23/addAddV2%lstm_23/lstm_cell_23/MatMul:product:0'lstm_23/lstm_cell_23/MatMul_1:product:0*
T0*'
_output_shapes
:         @ю
+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4lstm_23_lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0г
lstm_23/lstm_cell_23/BiasAddBiasAddlstm_23/lstm_cell_23/add:z:03lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @f
$lstm_23/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ш
lstm_23/lstm_cell_23/splitSplit-lstm_23/lstm_cell_23/split/split_dim:output:0%lstm_23/lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_split~
lstm_23/lstm_cell_23/SigmoidSigmoid#lstm_23/lstm_cell_23/split:output:0*
T0*'
_output_shapes
:         ђ
lstm_23/lstm_cell_23/Sigmoid_1Sigmoid#lstm_23/lstm_cell_23/split:output:1*
T0*'
_output_shapes
:         Ј
lstm_23/lstm_cell_23/mulMul"lstm_23/lstm_cell_23/Sigmoid_1:y:0lstm_23/zeros_1:output:0*
T0*'
_output_shapes
:         x
lstm_23/lstm_cell_23/ReluRelu#lstm_23/lstm_cell_23/split:output:2*
T0*'
_output_shapes
:         ъ
lstm_23/lstm_cell_23/mul_1Mul lstm_23/lstm_cell_23/Sigmoid:y:0'lstm_23/lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         Њ
lstm_23/lstm_cell_23/add_1AddV2lstm_23/lstm_cell_23/mul:z:0lstm_23/lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         ђ
lstm_23/lstm_cell_23/Sigmoid_2Sigmoid#lstm_23/lstm_cell_23/split:output:3*
T0*'
_output_shapes
:         u
lstm_23/lstm_cell_23/Relu_1Relulstm_23/lstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         б
lstm_23/lstm_cell_23/mul_2Mul"lstm_23/lstm_cell_23/Sigmoid_2:y:0)lstm_23/lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         v
%lstm_23/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       л
lstm_23/TensorArrayV2_1TensorListReserve.lstm_23/TensorArrayV2_1/element_shape:output:0 lstm_23/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмN
lstm_23/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_23/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         \
lstm_23/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ы
lstm_23/whileWhile#lstm_23/while/loop_counter:output:0)lstm_23/while/maximum_iterations:output:0lstm_23/time:output:0 lstm_23/TensorArrayV2_1:handle:0lstm_23/zeros:output:0lstm_23/zeros_1:output:0 lstm_23/strided_slice_1:output:0?lstm_23/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_23_lstm_cell_23_matmul_readvariableop_resource5lstm_23_lstm_cell_23_matmul_1_readvariableop_resource4lstm_23_lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_23_while_body_682974*%
condR
lstm_23_while_cond_682973*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ѕ
8lstm_23/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┌
*lstm_23/TensorArrayV2Stack/TensorListStackTensorListStacklstm_23/while:output:3Alstm_23/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0p
lstm_23/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         i
lstm_23/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_23/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:»
lstm_23/strided_slice_3StridedSlice3lstm_23/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_23/strided_slice_3/stack:output:0(lstm_23/strided_slice_3/stack_1:output:0(lstm_23/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskm
lstm_23/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
lstm_23/transpose_1	Transpose3lstm_23/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_23/transpose_1/perm:output:0*
T0*+
_output_shapes
:         c
lstm_23/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    s
dropout_35/IdentityIdentity lstm_23/strided_slice_3:output:0*
T0*'
_output_shapes
:         є
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Љ
dense_19/MatMulMatMuldropout_35/Identity:output:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         b
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*'
_output_shapes
:         n
dropout_36/IdentityIdentitydense_19/Relu:activations:0*
T0*'
_output_shapes
:         є
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Љ
dense_20/MatMulMatMuldropout_36/Identity:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         b
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*'
_output_shapes
:         n
dropout_37/IdentityIdentitydense_20/Relu:activations:0*
T0*'
_output_shapes
:         є
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Љ
dense_21/MatMulMatMuldropout_37/Identity:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         b
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*'
_output_shapes
:         n
dropout_38/IdentityIdentitydense_21/Relu:activations:0*
T0*'
_output_shapes
:         є
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Љ
dense_22/MatMulMatMuldropout_38/Identity:output:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
IdentityIdentitydense_22/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         а
NoOpNoOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp,^lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp+^lstm_21/lstm_cell_21/MatMul/ReadVariableOp-^lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp^lstm_21/while,^lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp+^lstm_22/lstm_cell_22/MatMul/ReadVariableOp-^lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp^lstm_22/while,^lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp+^lstm_23/lstm_cell_23/MatMul/ReadVariableOp-^lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp^lstm_23/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         : : : : : : : : : : : : : : : : : 2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2Z
+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp2X
*lstm_21/lstm_cell_21/MatMul/ReadVariableOp*lstm_21/lstm_cell_21/MatMul/ReadVariableOp2\
,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp2
lstm_21/whilelstm_21/while2Z
+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp2X
*lstm_22/lstm_cell_22/MatMul/ReadVariableOp*lstm_22/lstm_cell_22/MatMul/ReadVariableOp2\
,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp2
lstm_22/whilelstm_22/while2Z
+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp2X
*lstm_23/lstm_cell_23/MatMul/ReadVariableOp*lstm_23/lstm_cell_23/MatMul/ReadVariableOp2\
,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp2
lstm_23/whilelstm_23/while:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
 7
╩
while_body_685403
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_23_matmul_readvariableop_resource_0:@G
5while_lstm_cell_23_matmul_1_readvariableop_resource_0:@B
4while_lstm_cell_23_biasadd_readvariableop_resource_0:@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_23_matmul_readvariableop_resource:@E
3while_lstm_cell_23_matmul_1_readvariableop_resource:@@
2while_lstm_cell_23_biasadd_readvariableop_resource:@ѕб)while/lstm_cell_23/BiasAdd/ReadVariableOpб(while/lstm_cell_23/MatMul/ReadVariableOpб*while/lstm_cell_23/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ю
(while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_23_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype0╣
while/lstm_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @а
*while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_23_matmul_1_readvariableop_resource_0*
_output_shapes

:@*
dtype0а
while/lstm_cell_23/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ю
while/lstm_cell_23/addAddV2#while/lstm_cell_23/MatMul:product:0%while/lstm_cell_23/MatMul_1:product:0*
T0*'
_output_shapes
:         @џ
)while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0д
while/lstm_cell_23/BiasAddBiasAddwhile/lstm_cell_23/add:z:01while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
"while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_23/splitSplit+while/lstm_cell_23/split/split_dim:output:0#while/lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitz
while/lstm_cell_23/SigmoidSigmoid!while/lstm_cell_23/split:output:0*
T0*'
_output_shapes
:         |
while/lstm_cell_23/Sigmoid_1Sigmoid!while/lstm_cell_23/split:output:1*
T0*'
_output_shapes
:         є
while/lstm_cell_23/mulMul while/lstm_cell_23/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         t
while/lstm_cell_23/ReluRelu!while/lstm_cell_23/split:output:2*
T0*'
_output_shapes
:         ў
while/lstm_cell_23/mul_1Mulwhile/lstm_cell_23/Sigmoid:y:0%while/lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         Ї
while/lstm_cell_23/add_1AddV2while/lstm_cell_23/mul:z:0while/lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         |
while/lstm_cell_23/Sigmoid_2Sigmoid!while/lstm_cell_23/split:output:3*
T0*'
_output_shapes
:         q
while/lstm_cell_23/Relu_1Reluwhile/lstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         ю
while/lstm_cell_23/mul_2Mul while/lstm_cell_23/Sigmoid_2:y:0'while/lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         ┼
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_23/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_23/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         y
while/Identity_5Identitywhile/lstm_cell_23/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         л

while/NoOpNoOp*^while/lstm_cell_23/BiasAdd/ReadVariableOp)^while/lstm_cell_23/MatMul/ReadVariableOp+^while/lstm_cell_23/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_23_biasadd_readvariableop_resource4while_lstm_cell_23_biasadd_readvariableop_resource_0"l
3while_lstm_cell_23_matmul_1_readvariableop_resource5while_lstm_cell_23_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_23_matmul_readvariableop_resource3while_lstm_cell_23_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2V
)while/lstm_cell_23/BiasAdd/ReadVariableOp)while/lstm_cell_23/BiasAdd/ReadVariableOp2T
(while/lstm_cell_23/MatMul/ReadVariableOp(while/lstm_cell_23/MatMul/ReadVariableOp2X
*while/lstm_cell_23/MatMul_1/ReadVariableOp*while/lstm_cell_23/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
ќ

с
lstm_21_while_cond_683147,
(lstm_21_while_lstm_21_while_loop_counter2
.lstm_21_while_lstm_21_while_maximum_iterations
lstm_21_while_placeholder
lstm_21_while_placeholder_1
lstm_21_while_placeholder_2
lstm_21_while_placeholder_3.
*lstm_21_while_less_lstm_21_strided_slice_1D
@lstm_21_while_lstm_21_while_cond_683147___redundant_placeholder0D
@lstm_21_while_lstm_21_while_cond_683147___redundant_placeholder1D
@lstm_21_while_lstm_21_while_cond_683147___redundant_placeholder2D
@lstm_21_while_lstm_21_while_cond_683147___redundant_placeholder3
lstm_21_while_identity
ѓ
lstm_21/while/LessLesslstm_21_while_placeholder*lstm_21_while_less_lstm_21_strided_slice_1*
T0*
_output_shapes
: [
lstm_21/while/IdentityIdentitylstm_21/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_21_while_identitylstm_21/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
М
Ѓ
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_685838

inputs
states_0
states_10
matmul_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         Љ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
states/0:QM
'
_output_shapes
:         
"
_user_specified_name
states/1
ж
d
F__inference_dropout_34_layer_call_and_return_conditional_losses_684859

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         _

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         :S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╦
Ђ
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_680476

inputs

states
states_10
matmul_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         Љ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_namestates:OK
'
_output_shapes
:         
 
_user_specified_namestates
ЎJ
ў
C__inference_lstm_22_layer_call_and_return_conditional_losses_684844

inputs=
+lstm_cell_22_matmul_readvariableop_resource:@?
-lstm_cell_22_matmul_1_readvariableop_resource:@:
,lstm_cell_22_biasadd_readvariableop_resource:@
identityѕб#lstm_cell_22/BiasAdd/ReadVariableOpб"lstm_cell_22/MatMul/ReadVariableOpб$lstm_cell_22/MatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskј
"lstm_cell_22/MatMul/ReadVariableOpReadVariableOp+lstm_cell_22_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ћ
lstm_cell_22/MatMulMatMulstrided_slice_2:output:0*lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @њ
$lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_22_matmul_1_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
lstm_cell_22/MatMul_1MatMulzeros:output:0,lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @І
lstm_cell_22/addAddV2lstm_cell_22/MatMul:product:0lstm_cell_22/MatMul_1:product:0*
T0*'
_output_shapes
:         @ї
#lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ћ
lstm_cell_22/BiasAddBiasAddlstm_cell_22/add:z:0+lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @^
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0lstm_cell_22/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitn
lstm_cell_22/SigmoidSigmoidlstm_cell_22/split:output:0*
T0*'
_output_shapes
:         p
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/split:output:1*
T0*'
_output_shapes
:         w
lstm_cell_22/mulMullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         h
lstm_cell_22/ReluRelulstm_cell_22/split:output:2*
T0*'
_output_shapes
:         є
lstm_cell_22/mul_1Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Relu:activations:0*
T0*'
_output_shapes
:         {
lstm_cell_22/add_1AddV2lstm_cell_22/mul:z:0lstm_cell_22/mul_1:z:0*
T0*'
_output_shapes
:         p
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/split:output:3*
T0*'
_output_shapes
:         e
lstm_cell_22/Relu_1Relulstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         і
lstm_cell_22/mul_2Mullstm_cell_22/Sigmoid_2:y:0!lstm_cell_22/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѓ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_22_matmul_readvariableop_resource-lstm_cell_22_matmul_1_readvariableop_resource,lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_684760*
condR
while_cond_684759*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         └
NoOpNoOp$^lstm_cell_22/BiasAdd/ReadVariableOp#^lstm_cell_22/MatMul/ReadVariableOp%^lstm_cell_22/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2J
#lstm_cell_22/BiasAdd/ReadVariableOp#lstm_cell_22/BiasAdd/ReadVariableOp2H
"lstm_cell_22/MatMul/ReadVariableOp"lstm_cell_22/MatMul/ReadVariableOp2L
$lstm_cell_22/MatMul_1/ReadVariableOp$lstm_cell_22/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs"ѓL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╗
serving_defaultД
K
lstm_21_input:
serving_default_lstm_21_input:0         <
dense_220
StatefulPartitionedCall:0         tensorflow/serving/predict:чО
╦
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
Э__call__
+щ&call_and_return_all_conditional_losses
Щ_default_save_signature"
_tf_keras_sequential
┼
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
ч__call__
+Ч&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
Д
	variables
trainable_variables
regularization_losses
	keras_api
§__call__
+■&call_and_return_all_conditional_losses"
_tf_keras_layer
┼
cell

state_spec
 	variables
!trainable_variables
"regularization_losses
#	keras_api
 __call__
+ђ&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
Д
$	variables
%trainable_variables
&regularization_losses
'	keras_api
Ђ__call__
+ѓ&call_and_return_all_conditional_losses"
_tf_keras_layer
┼
(cell
)
state_spec
*	variables
+trainable_variables
,regularization_losses
-	keras_api
Ѓ__call__
+ё&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
Д
.	variables
/trainable_variables
0regularization_losses
1	keras_api
Ё__call__
+є&call_and_return_all_conditional_losses"
_tf_keras_layer
й

2kernel
3bias
4	variables
5trainable_variables
6regularization_losses
7	keras_api
Є__call__
+ѕ&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
8	variables
9trainable_variables
:regularization_losses
;	keras_api
Ѕ__call__
+і&call_and_return_all_conditional_losses"
_tf_keras_layer
й

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
І__call__
+ї&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
Ї__call__
+ј&call_and_return_all_conditional_losses"
_tf_keras_layer
й

Fkernel
Gbias
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
Ј__call__
+љ&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
Љ__call__
+њ&call_and_return_all_conditional_losses"
_tf_keras_layer
й

Pkernel
Qbias
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
Њ__call__
+ћ&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
Viter

Wbeta_1

Xbeta_2
	Ydecay
Zlearning_rate2mо3mО<mп=m┘Fm┌Gm█Pm▄QmП[mя\m▀]mЯ^mр_mР`mсamСbmтcmТ2vу3vУ<vж=vЖFvвGvВPvьQvЬ[v№\v­]vы^vЫ_vз`vЗavшbvШcvэ"
	optimizer
ъ
[0
\1
]2
^3
_4
`5
a6
b7
c8
29
310
<11
=12
F13
G14
P15
Q16"
trackable_list_wrapper
ъ
[0
\1
]2
^3
_4
`5
a6
b7
c8
29
310
<11
=12
F13
G14
P15
Q16"
trackable_list_wrapper
 "
trackable_list_wrapper
╬
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
regularization_losses
Э__call__
Щ_default_save_signature
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
-
Ћserving_default"
signature_map
с
i
state_size

[kernel
\recurrent_kernel
]bias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
ќ__call__
+Ќ&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
[0
\1
]2"
trackable_list_wrapper
5
[0
\1
]2"
trackable_list_wrapper
 "
trackable_list_wrapper
╝

nstates
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
	variables
trainable_variables
regularization_losses
ч__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
	variables
trainable_variables
regularization_losses
§__call__
+■&call_and_return_all_conditional_losses
'■"call_and_return_conditional_losses"
_generic_user_object
с
y
state_size

^kernel
_recurrent_kernel
`bias
z	variables
{trainable_variables
|regularization_losses
}	keras_api
ў__call__
+Ў&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
^0
_1
`2"
trackable_list_wrapper
5
^0
_1
`2"
trackable_list_wrapper
 "
trackable_list_wrapper
└

~states
non_trainable_variables
ђlayers
Ђmetrics
 ѓlayer_regularization_losses
Ѓlayer_metrics
 	variables
!trainable_variables
"regularization_losses
 __call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
ёnon_trainable_variables
Ёlayers
єmetrics
 Єlayer_regularization_losses
ѕlayer_metrics
$	variables
%trainable_variables
&regularization_losses
Ђ__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses"
_generic_user_object
У
Ѕ
state_size

akernel
brecurrent_kernel
cbias
і	variables
Іtrainable_variables
їregularization_losses
Ї	keras_api
џ__call__
+Џ&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
a0
b1
c2"
trackable_list_wrapper
5
a0
b1
c2"
trackable_list_wrapper
 "
trackable_list_wrapper
┬
јstates
Јnon_trainable_variables
љlayers
Љmetrics
 њlayer_regularization_losses
Њlayer_metrics
*	variables
+trainable_variables
,regularization_losses
Ѓ__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
ћnon_trainable_variables
Ћlayers
ќmetrics
 Ќlayer_regularization_losses
ўlayer_metrics
.	variables
/trainable_variables
0regularization_losses
Ё__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
!:2dense_19/kernel
:2dense_19/bias
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
х
Ўnon_trainable_variables
џlayers
Џmetrics
 юlayer_regularization_losses
Юlayer_metrics
4	variables
5trainable_variables
6regularization_losses
Є__call__
+ѕ&call_and_return_all_conditional_losses
'ѕ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
ъnon_trainable_variables
Ъlayers
аmetrics
 Аlayer_regularization_losses
бlayer_metrics
8	variables
9trainable_variables
:regularization_losses
Ѕ__call__
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses"
_generic_user_object
!:2dense_20/kernel
:2dense_20/bias
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
х
Бnon_trainable_variables
цlayers
Цmetrics
 дlayer_regularization_losses
Дlayer_metrics
>	variables
?trainable_variables
@regularization_losses
І__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
еnon_trainable_variables
Еlayers
фmetrics
 Фlayer_regularization_losses
гlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
Ї__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses"
_generic_user_object
!:2dense_21/kernel
:2dense_21/bias
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
х
Гnon_trainable_variables
«layers
»metrics
 ░layer_regularization_losses
▒layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
Ј__call__
+љ&call_and_return_all_conditional_losses
'љ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
▓non_trainable_variables
│layers
┤metrics
 хlayer_regularization_losses
Хlayer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
Љ__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses"
_generic_user_object
!:2dense_22/kernel
:2dense_22/bias
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
х
иnon_trainable_variables
Иlayers
╣metrics
 ║layer_regularization_losses
╗layer_metrics
R	variables
Strainable_variables
Tregularization_losses
Њ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
-:+@2lstm_21/lstm_cell_21/kernel
7:5@2%lstm_21/lstm_cell_21/recurrent_kernel
':%@2lstm_21/lstm_cell_21/bias
-:+@2lstm_22/lstm_cell_22/kernel
7:5@2%lstm_22/lstm_cell_22/recurrent_kernel
':%@2lstm_22/lstm_cell_22/bias
-:+@2lstm_23/lstm_cell_23/kernel
7:5@2%lstm_23/lstm_cell_23/recurrent_kernel
':%@2lstm_23/lstm_cell_23/bias
 "
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
0
╝0
й1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
[0
\1
]2"
trackable_list_wrapper
5
[0
\1
]2"
trackable_list_wrapper
 "
trackable_list_wrapper
х
Йnon_trainable_variables
┐layers
└metrics
 ┴layer_regularization_losses
┬layer_metrics
j	variables
ktrainable_variables
lregularization_losses
ќ__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
^0
_1
`2"
trackable_list_wrapper
5
^0
_1
`2"
trackable_list_wrapper
 "
trackable_list_wrapper
х
├non_trainable_variables
─layers
┼metrics
 кlayer_regularization_losses
Кlayer_metrics
z	variables
{trainable_variables
|regularization_losses
ў__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
a0
b1
c2"
trackable_list_wrapper
5
a0
b1
c2"
trackable_list_wrapper
 "
trackable_list_wrapper
И
╚non_trainable_variables
╔layers
╩metrics
 ╦layer_regularization_losses
╠layer_metrics
і	variables
Іtrainable_variables
їregularization_losses
џ__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
(0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

═total

╬count
¤	variables
л	keras_api"
_tf_keras_metric
c

Лtotal

мcount
М
_fn_kwargs
н	variables
Н	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
═0
╬1"
trackable_list_wrapper
.
¤	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Л0
м1"
trackable_list_wrapper
.
н	variables"
_generic_user_object
&:$2Adam/dense_19/kernel/m
 :2Adam/dense_19/bias/m
&:$2Adam/dense_20/kernel/m
 :2Adam/dense_20/bias/m
&:$2Adam/dense_21/kernel/m
 :2Adam/dense_21/bias/m
&:$2Adam/dense_22/kernel/m
 :2Adam/dense_22/bias/m
2:0@2"Adam/lstm_21/lstm_cell_21/kernel/m
<::@2,Adam/lstm_21/lstm_cell_21/recurrent_kernel/m
,:*@2 Adam/lstm_21/lstm_cell_21/bias/m
2:0@2"Adam/lstm_22/lstm_cell_22/kernel/m
<::@2,Adam/lstm_22/lstm_cell_22/recurrent_kernel/m
,:*@2 Adam/lstm_22/lstm_cell_22/bias/m
2:0@2"Adam/lstm_23/lstm_cell_23/kernel/m
<::@2,Adam/lstm_23/lstm_cell_23/recurrent_kernel/m
,:*@2 Adam/lstm_23/lstm_cell_23/bias/m
&:$2Adam/dense_19/kernel/v
 :2Adam/dense_19/bias/v
&:$2Adam/dense_20/kernel/v
 :2Adam/dense_20/bias/v
&:$2Adam/dense_21/kernel/v
 :2Adam/dense_21/bias/v
&:$2Adam/dense_22/kernel/v
 :2Adam/dense_22/bias/v
2:0@2"Adam/lstm_21/lstm_cell_21/kernel/v
<::@2,Adam/lstm_21/lstm_cell_21/recurrent_kernel/v
,:*@2 Adam/lstm_21/lstm_cell_21/bias/v
2:0@2"Adam/lstm_22/lstm_cell_22/kernel/v
<::@2,Adam/lstm_22/lstm_cell_22/recurrent_kernel/v
,:*@2 Adam/lstm_22/lstm_cell_22/bias/v
2:0@2"Adam/lstm_23/lstm_cell_23/kernel/v
<::@2,Adam/lstm_23/lstm_cell_23/recurrent_kernel/v
,:*@2 Adam/lstm_23/lstm_cell_23/bias/v
ѓ2 
-__inference_sequential_7_layer_call_fn_681567
-__inference_sequential_7_layer_call_fn_682596
-__inference_sequential_7_layer_call_fn_682635
-__inference_sequential_7_layer_call_fn_682408└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ь2в
H__inference_sequential_7_layer_call_and_return_conditional_losses_683089
H__inference_sequential_7_layer_call_and_return_conditional_losses_683585
H__inference_sequential_7_layer_call_and_return_conditional_losses_682459
H__inference_sequential_7_layer_call_and_return_conditional_losses_682510└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
мB¤
!__inference__wrapped_model_679913lstm_21_input"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ѓ2ђ
(__inference_lstm_21_layer_call_fn_683596
(__inference_lstm_21_layer_call_fn_683607
(__inference_lstm_21_layer_call_fn_683618
(__inference_lstm_21_layer_call_fn_683629Н
╠▓╚
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
№2В
C__inference_lstm_21_layer_call_and_return_conditional_losses_683772
C__inference_lstm_21_layer_call_and_return_conditional_losses_683915
C__inference_lstm_21_layer_call_and_return_conditional_losses_684058
C__inference_lstm_21_layer_call_and_return_conditional_losses_684201Н
╠▓╚
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ћ2Љ
+__inference_dropout_33_layer_call_fn_684206
+__inference_dropout_33_layer_call_fn_684211┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╩2К
F__inference_dropout_33_layer_call_and_return_conditional_losses_684216
F__inference_dropout_33_layer_call_and_return_conditional_losses_684228┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ѓ2ђ
(__inference_lstm_22_layer_call_fn_684239
(__inference_lstm_22_layer_call_fn_684250
(__inference_lstm_22_layer_call_fn_684261
(__inference_lstm_22_layer_call_fn_684272Н
╠▓╚
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
№2В
C__inference_lstm_22_layer_call_and_return_conditional_losses_684415
C__inference_lstm_22_layer_call_and_return_conditional_losses_684558
C__inference_lstm_22_layer_call_and_return_conditional_losses_684701
C__inference_lstm_22_layer_call_and_return_conditional_losses_684844Н
╠▓╚
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ћ2Љ
+__inference_dropout_34_layer_call_fn_684849
+__inference_dropout_34_layer_call_fn_684854┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╩2К
F__inference_dropout_34_layer_call_and_return_conditional_losses_684859
F__inference_dropout_34_layer_call_and_return_conditional_losses_684871┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ѓ2ђ
(__inference_lstm_23_layer_call_fn_684882
(__inference_lstm_23_layer_call_fn_684893
(__inference_lstm_23_layer_call_fn_684904
(__inference_lstm_23_layer_call_fn_684915Н
╠▓╚
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
№2В
C__inference_lstm_23_layer_call_and_return_conditional_losses_685058
C__inference_lstm_23_layer_call_and_return_conditional_losses_685201
C__inference_lstm_23_layer_call_and_return_conditional_losses_685344
C__inference_lstm_23_layer_call_and_return_conditional_losses_685487Н
╠▓╚
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ћ2Љ
+__inference_dropout_35_layer_call_fn_685492
+__inference_dropout_35_layer_call_fn_685497┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╩2К
F__inference_dropout_35_layer_call_and_return_conditional_losses_685502
F__inference_dropout_35_layer_call_and_return_conditional_losses_685514┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
М2л
)__inference_dense_19_layer_call_fn_685523б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_dense_19_layer_call_and_return_conditional_losses_685534б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћ2Љ
+__inference_dropout_36_layer_call_fn_685539
+__inference_dropout_36_layer_call_fn_685544┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╩2К
F__inference_dropout_36_layer_call_and_return_conditional_losses_685549
F__inference_dropout_36_layer_call_and_return_conditional_losses_685561┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
М2л
)__inference_dense_20_layer_call_fn_685570б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_dense_20_layer_call_and_return_conditional_losses_685581б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћ2Љ
+__inference_dropout_37_layer_call_fn_685586
+__inference_dropout_37_layer_call_fn_685591┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╩2К
F__inference_dropout_37_layer_call_and_return_conditional_losses_685596
F__inference_dropout_37_layer_call_and_return_conditional_losses_685608┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
М2л
)__inference_dense_21_layer_call_fn_685617б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_dense_21_layer_call_and_return_conditional_losses_685628б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћ2Љ
+__inference_dropout_38_layer_call_fn_685633
+__inference_dropout_38_layer_call_fn_685638┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╩2К
F__inference_dropout_38_layer_call_and_return_conditional_losses_685643
F__inference_dropout_38_layer_call_and_return_conditional_losses_685655┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
М2л
)__inference_dense_22_layer_call_fn_685664б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_dense_22_layer_call_and_return_conditional_losses_685674б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЛB╬
$__inference_signature_wrapper_682557lstm_21_input"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
б2Ъ
-__inference_lstm_cell_21_layer_call_fn_685691
-__inference_lstm_cell_21_layer_call_fn_685708Й
х▓▒
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
п2Н
H__inference_lstm_cell_21_layer_call_and_return_conditional_losses_685740
H__inference_lstm_cell_21_layer_call_and_return_conditional_losses_685772Й
х▓▒
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
б2Ъ
-__inference_lstm_cell_22_layer_call_fn_685789
-__inference_lstm_cell_22_layer_call_fn_685806Й
х▓▒
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
п2Н
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_685838
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_685870Й
х▓▒
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
б2Ъ
-__inference_lstm_cell_23_layer_call_fn_685887
-__inference_lstm_cell_23_layer_call_fn_685904Й
х▓▒
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
п2Н
H__inference_lstm_cell_23_layer_call_and_return_conditional_losses_685936
H__inference_lstm_cell_23_layer_call_and_return_conditional_losses_685968Й
х▓▒
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 ф
!__inference__wrapped_model_679913ё[\]^_`abc23<=FGPQ:б7
0б-
+і(
lstm_21_input         
ф "3ф0
.
dense_22"і
dense_22         ц
D__inference_dense_19_layer_call_and_return_conditional_losses_685534\23/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ |
)__inference_dense_19_layer_call_fn_685523O23/б,
%б"
 і
inputs         
ф "і         ц
D__inference_dense_20_layer_call_and_return_conditional_losses_685581\<=/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ |
)__inference_dense_20_layer_call_fn_685570O<=/б,
%б"
 і
inputs         
ф "і         ц
D__inference_dense_21_layer_call_and_return_conditional_losses_685628\FG/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ |
)__inference_dense_21_layer_call_fn_685617OFG/б,
%б"
 і
inputs         
ф "і         ц
D__inference_dense_22_layer_call_and_return_conditional_losses_685674\PQ/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ |
)__inference_dense_22_layer_call_fn_685664OPQ/б,
%б"
 і
inputs         
ф "і         «
F__inference_dropout_33_layer_call_and_return_conditional_losses_684216d7б4
-б*
$і!
inputs         
p 
ф ")б&
і
0         
џ «
F__inference_dropout_33_layer_call_and_return_conditional_losses_684228d7б4
-б*
$і!
inputs         
p
ф ")б&
і
0         
џ є
+__inference_dropout_33_layer_call_fn_684206W7б4
-б*
$і!
inputs         
p 
ф "і         є
+__inference_dropout_33_layer_call_fn_684211W7б4
-б*
$і!
inputs         
p
ф "і         «
F__inference_dropout_34_layer_call_and_return_conditional_losses_684859d7б4
-б*
$і!
inputs         
p 
ф ")б&
і
0         
џ «
F__inference_dropout_34_layer_call_and_return_conditional_losses_684871d7б4
-б*
$і!
inputs         
p
ф ")б&
і
0         
џ є
+__inference_dropout_34_layer_call_fn_684849W7б4
-б*
$і!
inputs         
p 
ф "і         є
+__inference_dropout_34_layer_call_fn_684854W7б4
-б*
$і!
inputs         
p
ф "і         д
F__inference_dropout_35_layer_call_and_return_conditional_losses_685502\3б0
)б&
 і
inputs         
p 
ф "%б"
і
0         
џ д
F__inference_dropout_35_layer_call_and_return_conditional_losses_685514\3б0
)б&
 і
inputs         
p
ф "%б"
і
0         
џ ~
+__inference_dropout_35_layer_call_fn_685492O3б0
)б&
 і
inputs         
p 
ф "і         ~
+__inference_dropout_35_layer_call_fn_685497O3б0
)б&
 і
inputs         
p
ф "і         д
F__inference_dropout_36_layer_call_and_return_conditional_losses_685549\3б0
)б&
 і
inputs         
p 
ф "%б"
і
0         
џ д
F__inference_dropout_36_layer_call_and_return_conditional_losses_685561\3б0
)б&
 і
inputs         
p
ф "%б"
і
0         
џ ~
+__inference_dropout_36_layer_call_fn_685539O3б0
)б&
 і
inputs         
p 
ф "і         ~
+__inference_dropout_36_layer_call_fn_685544O3б0
)б&
 і
inputs         
p
ф "і         д
F__inference_dropout_37_layer_call_and_return_conditional_losses_685596\3б0
)б&
 і
inputs         
p 
ф "%б"
і
0         
џ д
F__inference_dropout_37_layer_call_and_return_conditional_losses_685608\3б0
)б&
 і
inputs         
p
ф "%б"
і
0         
џ ~
+__inference_dropout_37_layer_call_fn_685586O3б0
)б&
 і
inputs         
p 
ф "і         ~
+__inference_dropout_37_layer_call_fn_685591O3б0
)б&
 і
inputs         
p
ф "і         д
F__inference_dropout_38_layer_call_and_return_conditional_losses_685643\3б0
)б&
 і
inputs         
p 
ф "%б"
і
0         
џ д
F__inference_dropout_38_layer_call_and_return_conditional_losses_685655\3б0
)б&
 і
inputs         
p
ф "%б"
і
0         
џ ~
+__inference_dropout_38_layer_call_fn_685633O3б0
)б&
 і
inputs         
p 
ф "і         ~
+__inference_dropout_38_layer_call_fn_685638O3б0
)б&
 і
inputs         
p
ф "і         м
C__inference_lstm_21_layer_call_and_return_conditional_losses_683772і[\]OбL
EбB
4џ1
/і,
inputs/0                  

 
p 

 
ф "2б/
(і%
0                  
џ м
C__inference_lstm_21_layer_call_and_return_conditional_losses_683915і[\]OбL
EбB
4џ1
/і,
inputs/0                  

 
p

 
ф "2б/
(і%
0                  
џ И
C__inference_lstm_21_layer_call_and_return_conditional_losses_684058q[\]?б<
5б2
$і!
inputs         

 
p 

 
ф ")б&
і
0         
џ И
C__inference_lstm_21_layer_call_and_return_conditional_losses_684201q[\]?б<
5б2
$і!
inputs         

 
p

 
ф ")б&
і
0         
џ Е
(__inference_lstm_21_layer_call_fn_683596}[\]OбL
EбB
4џ1
/і,
inputs/0                  

 
p 

 
ф "%і"                  Е
(__inference_lstm_21_layer_call_fn_683607}[\]OбL
EбB
4џ1
/і,
inputs/0                  

 
p

 
ф "%і"                  љ
(__inference_lstm_21_layer_call_fn_683618d[\]?б<
5б2
$і!
inputs         

 
p 

 
ф "і         љ
(__inference_lstm_21_layer_call_fn_683629d[\]?б<
5б2
$і!
inputs         

 
p

 
ф "і         м
C__inference_lstm_22_layer_call_and_return_conditional_losses_684415і^_`OбL
EбB
4џ1
/і,
inputs/0                  

 
p 

 
ф "2б/
(і%
0                  
џ м
C__inference_lstm_22_layer_call_and_return_conditional_losses_684558і^_`OбL
EбB
4џ1
/і,
inputs/0                  

 
p

 
ф "2б/
(і%
0                  
џ И
C__inference_lstm_22_layer_call_and_return_conditional_losses_684701q^_`?б<
5б2
$і!
inputs         

 
p 

 
ф ")б&
і
0         
џ И
C__inference_lstm_22_layer_call_and_return_conditional_losses_684844q^_`?б<
5б2
$і!
inputs         

 
p

 
ф ")б&
і
0         
џ Е
(__inference_lstm_22_layer_call_fn_684239}^_`OбL
EбB
4џ1
/і,
inputs/0                  

 
p 

 
ф "%і"                  Е
(__inference_lstm_22_layer_call_fn_684250}^_`OбL
EбB
4џ1
/і,
inputs/0                  

 
p

 
ф "%і"                  љ
(__inference_lstm_22_layer_call_fn_684261d^_`?б<
5б2
$і!
inputs         

 
p 

 
ф "і         љ
(__inference_lstm_22_layer_call_fn_684272d^_`?б<
5б2
$і!
inputs         

 
p

 
ф "і         ─
C__inference_lstm_23_layer_call_and_return_conditional_losses_685058}abcOбL
EбB
4џ1
/і,
inputs/0                  

 
p 

 
ф "%б"
і
0         
џ ─
C__inference_lstm_23_layer_call_and_return_conditional_losses_685201}abcOбL
EбB
4џ1
/і,
inputs/0                  

 
p

 
ф "%б"
і
0         
џ ┤
C__inference_lstm_23_layer_call_and_return_conditional_losses_685344mabc?б<
5б2
$і!
inputs         

 
p 

 
ф "%б"
і
0         
џ ┤
C__inference_lstm_23_layer_call_and_return_conditional_losses_685487mabc?б<
5б2
$і!
inputs         

 
p

 
ф "%б"
і
0         
џ ю
(__inference_lstm_23_layer_call_fn_684882pabcOбL
EбB
4џ1
/і,
inputs/0                  

 
p 

 
ф "і         ю
(__inference_lstm_23_layer_call_fn_684893pabcOбL
EбB
4џ1
/і,
inputs/0                  

 
p

 
ф "і         ї
(__inference_lstm_23_layer_call_fn_684904`abc?б<
5б2
$і!
inputs         

 
p 

 
ф "і         ї
(__inference_lstm_23_layer_call_fn_684915`abc?б<
5б2
$і!
inputs         

 
p

 
ф "і         ╩
H__inference_lstm_cell_21_layer_call_and_return_conditional_losses_685740§[\]ђб}
vбs
 і
inputs         
KбH
"і
states/0         
"і
states/1         
p 
ф "sбp
iбf
і
0/0         
EџB
і
0/1/0         
і
0/1/1         
џ ╩
H__inference_lstm_cell_21_layer_call_and_return_conditional_losses_685772§[\]ђб}
vбs
 і
inputs         
KбH
"і
states/0         
"і
states/1         
p
ф "sбp
iбf
і
0/0         
EџB
і
0/1/0         
і
0/1/1         
џ Ъ
-__inference_lstm_cell_21_layer_call_fn_685691ь[\]ђб}
vбs
 і
inputs         
KбH
"і
states/0         
"і
states/1         
p 
ф "cб`
і
0         
Aџ>
і
1/0         
і
1/1         Ъ
-__inference_lstm_cell_21_layer_call_fn_685708ь[\]ђб}
vбs
 і
inputs         
KбH
"і
states/0         
"і
states/1         
p
ф "cб`
і
0         
Aџ>
і
1/0         
і
1/1         ╩
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_685838§^_`ђб}
vбs
 і
inputs         
KбH
"і
states/0         
"і
states/1         
p 
ф "sбp
iбf
і
0/0         
EџB
і
0/1/0         
і
0/1/1         
џ ╩
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_685870§^_`ђб}
vбs
 і
inputs         
KбH
"і
states/0         
"і
states/1         
p
ф "sбp
iбf
і
0/0         
EџB
і
0/1/0         
і
0/1/1         
џ Ъ
-__inference_lstm_cell_22_layer_call_fn_685789ь^_`ђб}
vбs
 і
inputs         
KбH
"і
states/0         
"і
states/1         
p 
ф "cб`
і
0         
Aџ>
і
1/0         
і
1/1         Ъ
-__inference_lstm_cell_22_layer_call_fn_685806ь^_`ђб}
vбs
 і
inputs         
KбH
"і
states/0         
"і
states/1         
p
ф "cб`
і
0         
Aџ>
і
1/0         
і
1/1         ╩
H__inference_lstm_cell_23_layer_call_and_return_conditional_losses_685936§abcђб}
vбs
 і
inputs         
KбH
"і
states/0         
"і
states/1         
p 
ф "sбp
iбf
і
0/0         
EџB
і
0/1/0         
і
0/1/1         
џ ╩
H__inference_lstm_cell_23_layer_call_and_return_conditional_losses_685968§abcђб}
vбs
 і
inputs         
KбH
"і
states/0         
"і
states/1         
p
ф "sбp
iбf
і
0/0         
EџB
і
0/1/0         
і
0/1/1         
џ Ъ
-__inference_lstm_cell_23_layer_call_fn_685887ьabcђб}
vбs
 і
inputs         
KбH
"і
states/0         
"і
states/1         
p 
ф "cб`
і
0         
Aџ>
і
1/0         
і
1/1         Ъ
-__inference_lstm_cell_23_layer_call_fn_685904ьabcђб}
vбs
 і
inputs         
KбH
"і
states/0         
"і
states/1         
p
ф "cб`
і
0         
Aџ>
і
1/0         
і
1/1         ╩
H__inference_sequential_7_layer_call_and_return_conditional_losses_682459~[\]^_`abc23<=FGPQBб?
8б5
+і(
lstm_21_input         
p 

 
ф "%б"
і
0         
џ ╩
H__inference_sequential_7_layer_call_and_return_conditional_losses_682510~[\]^_`abc23<=FGPQBб?
8б5
+і(
lstm_21_input         
p

 
ф "%б"
і
0         
џ ├
H__inference_sequential_7_layer_call_and_return_conditional_losses_683089w[\]^_`abc23<=FGPQ;б8
1б.
$і!
inputs         
p 

 
ф "%б"
і
0         
џ ├
H__inference_sequential_7_layer_call_and_return_conditional_losses_683585w[\]^_`abc23<=FGPQ;б8
1б.
$і!
inputs         
p

 
ф "%б"
і
0         
џ б
-__inference_sequential_7_layer_call_fn_681567q[\]^_`abc23<=FGPQBб?
8б5
+і(
lstm_21_input         
p 

 
ф "і         б
-__inference_sequential_7_layer_call_fn_682408q[\]^_`abc23<=FGPQBб?
8б5
+і(
lstm_21_input         
p

 
ф "і         Џ
-__inference_sequential_7_layer_call_fn_682596j[\]^_`abc23<=FGPQ;б8
1б.
$і!
inputs         
p 

 
ф "і         Џ
-__inference_sequential_7_layer_call_fn_682635j[\]^_`abc23<=FGPQ;б8
1б.
$і!
inputs         
p

 
ф "і         Й
$__inference_signature_wrapper_682557Ћ[\]^_`abc23<=FGPQKбH
б 
Aф>
<
lstm_21_input+і(
lstm_21_input         "3ф0
.
dense_22"і
dense_22         