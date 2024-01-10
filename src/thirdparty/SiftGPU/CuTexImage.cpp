////////////////////////////////////////////////////////////////////////////
//	File:		CuTexImage.cpp
//	Author:		Changchang Wu
//	Description : implementation of the CuTexImage class.
//
//	Copyright (c) 2007 University of North Carolina at Chapel Hill
//	All Rights Reserved
//
//	Permission to use, copy, modify and distribute this software and its
//	documentation for educational, research and non-profit purposes, without
//	fee, and without a written agreement is hereby granted, provided that the
//	above copyright notice and the following paragraph appear in all copies.
//
//	The University of North Carolina at Chapel Hill make no representations
//	about the suitability of this software for any purpose. It is provided
//	'as is' without express or implied warranty.
//
//	Please send BUG REPORTS to ccwu@cs.unc.edu
//
////////////////////////////////////////////////////////////////////////////

#if defined(CUDA_SIFTGPU_ENABLED)

#include "GL/glew.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <cstring>
using namespace std;


#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_gl_interop.h>

#include "GlobalUtil.h"
#include "GLTexImage.h"
#include "CuTexImage.h"
#include "ProgramCU.h"

CuTexImage::CuTexObj::~CuTexObj()
{
	hipDestroyTextureObject(handle);
}

CuTexImage::CuTexObj CuTexImage::BindTexture(const hipTextureDesc& textureDesc,
											   										 const hipChannelFormatDesc& channelFmtDesc)
{
	CuTexObj texObj;

	hipResourceDesc resourceDesc;
	memset(&resourceDesc, 0, sizeof(resourceDesc));
  resourceDesc.resType = hipResourceTypeLinear;
  resourceDesc.res.linear.devPtr = _cuData;
	resourceDesc.res.linear.desc = channelFmtDesc;
	resourceDesc.res.linear.sizeInBytes = _numBytes;

	hipCreateTextureObject(&texObj.handle, &resourceDesc, &textureDesc, nullptr);
	ProgramCU::CheckErrorCUDA("CuTexImage::BindTexture");

	return texObj;
}

CuTexImage::CuTexObj CuTexImage::BindTexture2D(const hipTextureDesc& textureDesc,
											   											 const hipChannelFormatDesc& channelFmtDesc)
{
	CuTexObj texObj;

	hipResourceDesc resourceDesc;
	memset(&resourceDesc, 0, sizeof(resourceDesc));
	resourceDesc.resType = hipResourceTypePitch2D;
  resourceDesc.res.pitch2D.devPtr = _cuData;
	resourceDesc.res.pitch2D.width = _imgWidth;
	resourceDesc.res.pitch2D.height = _imgHeight;
	resourceDesc.res.pitch2D.pitchInBytes = _imgWidth * _numChannel * sizeof(float);
	resourceDesc.res.pitch2D.desc = channelFmtDesc;

	hipCreateTextureObject(&texObj.handle, &resourceDesc, &textureDesc, nullptr);
	ProgramCU::CheckErrorCUDA("CuTexImage::BindTexture2D");

	return texObj;
}

CuTexImage::CuTexImage()
{
	_cuData = NULL;
	_cuData2D = NULL;
	_fromPBO = 0;
	_numChannel = _numBytes = 0;
	_imgWidth = _imgHeight = _texWidth = _texHeight = 0;
}

CuTexImage::CuTexImage(int width, int height, int nchannel, GLuint pbo)
{
	_cuData = NULL;

	//check size of pbo
	GLint bsize, esize = width * height * nchannel * sizeof(float);
	glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, pbo);
	glGetBufferParameteriv(GL_PIXEL_PACK_BUFFER_ARB, GL_BUFFER_SIZE, &bsize);
	if(bsize < esize)
	{
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB, esize,	NULL, GL_STATIC_DRAW_ARB);
		glGetBufferParameteriv(GL_PIXEL_PACK_BUFFER_ARB, GL_BUFFER_SIZE, &bsize);
	}
	glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
	if(bsize >=esize)
	{

    hipGraphicsGLRegisterBuffer(&_cuData, pbo, hipGraphicsRegisterFlagsNone);
		ProgramCU::CheckErrorCUDA("hipGraphicsGLRegisterBuffer");
		_fromPBO = pbo;
	}else
	{
		_cuData = NULL;
		_fromPBO = 0;
	}
	if(_cuData)
	{
		_numBytes = bsize;
		_imgWidth = width;
		_imgHeight = height;
		_numChannel = nchannel;
	}else
	{
		_numBytes = 0;
		_imgWidth = 0;
		_imgHeight = 0;
		_numChannel = 0;
	}

	_texWidth = _texHeight =0;

	_cuData2D = NULL;
}

CuTexImage::~CuTexImage()
{


	if(_fromPBO)
	{
		//cudaGLUnmapBufferObject(_fromPBO);
		//cudaGLUnregisterBufferObject(_fromPBO);
	}else if(_cuData)
	{
		hipFree(_cuData);
	}
	if(_cuData2D)  hipFreeArray(_cuData2D);
}

void CuTexImage::SetImageSize(int width, int height)
{
	_imgWidth = width;
	_imgHeight = height;
}

bool CuTexImage::InitTexture(int width, int height, int nchannel)
{
	_imgWidth = width;
	_imgHeight = height;
	_numChannel = min(max(nchannel, 1), 4);

	const size_t size = width * height * _numChannel * sizeof(float);

  if (size < 0) {
    return false;
  }

  // SiftGPU uses int for all indexes and
  // this ensures that all elements can be accessed.
  if (size >= INT_MAX * sizeof(float)) {
    return false;
  }

	if(size <= _numBytes) return true;

	if(_cuData) hipFree(_cuData);

	//allocate the array data
	const hipError_t status = hipMalloc(&_cuData, _numBytes = size);

  if (status != hipSuccess) {
    _cuData = NULL;
    _numBytes = 0;
    return false;
  }

  return true;
}

void CuTexImage::CopyFromHost(const void * buf)
{
	if(_cuData == NULL) return;
	hipMemcpy( _cuData, buf, _imgWidth * _imgHeight * _numChannel * sizeof(float), hipMemcpyHostToDevice);
}

void CuTexImage::CopyToHost(void * buf)
{
	if(_cuData == NULL) return;
	hipMemcpy(buf, _cuData, _imgWidth * _imgHeight * _numChannel * sizeof(float), hipMemcpyDeviceToHost);
}

void CuTexImage::CopyToHost(void * buf, int stream)
{
	if(_cuData == NULL) return;
	hipMemcpyAsync(buf, _cuData, _imgWidth * _imgHeight * _numChannel * sizeof(float), hipMemcpyDeviceToHost, (hipStream_t)stream);
}

void CuTexImage::CopyFromPBO(int width, int height, GLuint pbo)
{
	hipGraphicsResource* pbuf =NULL;
	GLint esize = width * height * sizeof(float);
  hipGraphicsGLRegisterBuffer(&pbuf, pbo, hipGraphicsRegisterFlagsWriteDiscard);

	hipMemcpy(_cuData, pbuf, esize, hipMemcpyDeviceToDevice);
}

int CuTexImage::CopyToPBO(GLuint pbo)
{
	hipGraphicsResource* pbuf =NULL;
	GLint bsize, esize = _imgWidth * _imgHeight * sizeof(float) * _numChannel;
	glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, pbo);
	glGetBufferParameteriv(GL_PIXEL_PACK_BUFFER_ARB, GL_BUFFER_SIZE, &bsize);
	if(bsize < esize)
	{
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB, esize*3/2,	NULL, GL_STATIC_DRAW_ARB);
		glGetBufferParameteriv(GL_PIXEL_PACK_BUFFER_ARB, GL_BUFFER_SIZE, &bsize);
	}
	glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);

	if(bsize >= esize)
	{
    hipGraphicsGLRegisterBuffer(&pbuf, pbo, hipGraphicsRegisterFlagsWriteDiscard);
		hipMemcpy(pbuf, _cuData, esize, hipMemcpyDeviceToDevice);
		return 1;
	}else
	{
		return 0;
	}
}

#endif

