TARGET = .\bin\mp5.2

SRC_DIR = .\src
SRC_SUBDIR += .
OBJ_DIR = .\obj

LIBWB_SRC_DIR += ..\libwb
LIBWBS_SUBDIR += .
LIBWB_OBJ_DIR += ..\libwb\obj


CC = nvcc
C_FLAGS = -g -std=c++11 -O3
LD = $(CC)
INCLUDES += -I$(INCLUDE_DIR)
LD_FLAFS += 
LD_LIBS =

RM = del


SRCS += ${foreach subdir, $(SRC_SUBDIR), ${wildcard $(SRC_DIR)/$(subdir)/*.cu}}
OBJS += ${foreach src, $(notdir $(SRCS)), ${patsubst %.cu, $(OBJ_DIR)/%.obj, $(src)}}

LIBWB_SRCS += ${foreach subdir, $(LIBWBS_SUBDIR), ${wildcard $(LIBWB_SRC_DIR)/$(subdir)/*.cpp}}
OBJS += ${foreach src, $(notdir $(LIBWB_SRCS)), ${patsubst %.cpp, $(LIBWB_OBJ_DIR)/%.obj, $(src)}}


all: $(TARGET)
	@echo Builded target: $^
	@echo Done

$(TARGET): $(OBJS)
	@if not exist "$(@D)" mkdir "$(@D)"
	@echo Linking $@ from $^ ...
	nvcc -o $@.exe $^
	@echo Link finished

$(OBJ_DIR)/%.obj: $(SRC_DIR)/%.cu
	@if not exist "$(@D)" mkdir "$(@D)"
	@echo Compiling $@ from $< ...
	nvcc -I $(LIBWB_SRC_DIR) -c $< -o $@
	@echo Compile finished

$(LIBWB_OBJ_DIR)/%.obj: $(LIBWB_SRC_DIR)/%.cpp
	@if not exist "$(@D)" mkdir "$(@D)"
	@echo Compiling $@ from $< ...
	nvcc -c $< -o $@
	@echo Compile finished


.PHONY: clean cleanAll cleanObj cleanBinary 

clean: cleanBinary cleanobj 
cleanAll: cleanBinary cleanobj cleanLibwb

cleanBinary:
	@echo Remove all executable files'
	@-$(RM) "$(TARGET).exe"
	@-$(RM) "$(TARGET).exp"
	@-$(RM) "$(TARGET).lib"
cleanobj:
	@echo Remove object files
	@-$(RM) "$(OBJ_DIR)\*.obj"
cleanLibwb:
	@echo Remove libwb object files
	@-$(RM) "$(LIBWB_OBJ_DIR)\*.obj"


case%:
	@$(TARGET).exe -i .\data\$(patsubst case%,%,$@)\input.raw -e .\data\$(patsubst case%,%,$@)\output.raw -t vector



