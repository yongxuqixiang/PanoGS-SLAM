/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& mask,
	const torch::Tensor& gaussian_mask,
	const bool prefiltered,
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  const float* mask_ptr = nullptr;
  if (mask.defined() && mask.numel() != 0) {
    TORCH_CHECK(
        mask.device() == means3D.device(),
        "mask tensor must be on the same device as means3D");
    TORCH_CHECK(
        mask.scalar_type() == torch::kFloat32,
        "mask tensor must have dtype float32");
    TORCH_CHECK(
        mask.is_contiguous(),
        "mask tensor must be contiguous");
    TORCH_CHECK(
        (mask.dim() == 2 && mask.size(0) == H && mask.size(1) == W) ||
        (mask.dim() == 3 && mask.size(0) == 1 && mask.size(1) == H && mask.size(2) == W),
        "mask tensor must have shape (H, W) or (1, H, W)");
    mask_ptr = mask.data_ptr<float>();
  }
  
  const float* gaussian_mask_ptr = nullptr;
  if (gaussian_mask.defined() && gaussian_mask.numel() != 0) {
    TORCH_CHECK(
        gaussian_mask.device() == means3D.device(),
        "gaussian_mask tensor must be on the same device as means3D");
    TORCH_CHECK(
        gaussian_mask.scalar_type() == torch::kFloat32,
        "gaussian_mask tensor must have dtype float32");
    TORCH_CHECK(
        gaussian_mask.is_contiguous(),
        "gaussian_mask tensor must be contiguous");
    TORCH_CHECK(
        gaussian_mask.dim() == 1 && gaussian_mask.size(0) == P,
        "gaussian_mask tensor must have shape (P,)");
    gaussian_mask_ptr = gaussian_mask.data_ptr<float>();
  }

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_depth = torch::full({H, W}, 0.0, float_opts);
  torch::Tensor out_acc = torch::full({H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor n_touched = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor out_opaticy = torch::full({1, H, W}, 0.0, float_opts);


  torch::Tensor psi = torch::full({P}, 0, means3D.options().dtype(torch::kFloat32));
  torch::Tensor lat = torch::full({P}, 0, means3D.options().dtype(torch::kFloat32));
  torch::Tensor lon = torch::full({P}, 0, means3D.options().dtype(torch::kFloat32));

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		campos.contiguous().data<float>(),
		mask_ptr,
		gaussian_mask_ptr,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_depth.contiguous().data<float>(),
		out_acc.contiguous().data<float>(),
		out_opaticy.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		psi.contiguous().data<float>(),
		lat.contiguous().data<float>(),
		lon.contiguous().data<float>(),
		n_touched.contiguous().data<int>(),
		debug);
  }
  return std::make_tuple(rendered, out_color, out_depth, out_acc, radii, psi, lat, lon, geomBuffer, binningBuffer, imgBuffer, out_opaticy, n_touched);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors_precomp,
	const torch::Tensor& accum_depth,
	const torch::Tensor& accum_acc,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_depth,
	const torch::Tensor& dL_dout_acc,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_ddepths = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dtau = torch::zeros({P,6}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H,
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors_precomp.contiguous().data<float>(),
	  accum_depth.contiguous().data<float>(),
	  accum_acc.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_depth.contiguous().data<float>(),
	  dL_dout_acc.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_ddepths.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
      dL_dtau.contiguous().data<float>(),
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations, dL_dtau);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}