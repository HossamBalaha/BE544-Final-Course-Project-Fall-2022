-- phpMyAdmin SQL Dump
-- version 5.2.0
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Nov 29, 2022 at 09:15 PM
-- Server version: 10.4.27-MariaDB
-- PHP Version: 8.1.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `be544`
--

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `id` bigint(20) UNSIGNED NOT NULL,
  `username` varchar(100) NOT NULL,
  `password` varchar(100) NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `user_augmentations`
--

CREATE TABLE `user_augmentations` (
  `id` bigint(20) NOT NULL,
  `user_id` bigint(20) UNSIGNED DEFAULT NULL,
  `name` text NOT NULL,
  `store_name` text NOT NULL,
  `configurations` text DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `user_classifications`
--

CREATE TABLE `user_classifications` (
  `id` bigint(20) UNSIGNED NOT NULL,
  `user_id` bigint(20) UNSIGNED DEFAULT NULL,
  `name` text NOT NULL,
  `store_name` text NOT NULL,
  `relative_path` text NOT NULL,
  `configurations` text NOT NULL,
  `post_configurations` text DEFAULT NULL,
  `is_success` tinyint(1) NOT NULL DEFAULT 0,
  `message` text DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `user_datasets`
--

CREATE TABLE `user_datasets` (
  `id` bigint(20) UNSIGNED NOT NULL,
  `user_id` bigint(20) UNSIGNED DEFAULT NULL,
  `name` text NOT NULL,
  `store_name` text NOT NULL,
  `relative_path` text NOT NULL,
  `is_file` tinyint(1) NOT NULL DEFAULT 0,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `user_segmentations`
--

CREATE TABLE `user_segmentations` (
  `id` bigint(20) UNSIGNED NOT NULL,
  `user_id` bigint(20) UNSIGNED DEFAULT NULL,
  `name` text NOT NULL,
  `store_name` text NOT NULL,
  `relative_path` text NOT NULL,
  `configurations` text NOT NULL,
  `post_configurations` text DEFAULT NULL,
  `is_success` tinyint(1) NOT NULL DEFAULT 0,
  `message` text DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `user_wsis`
--

CREATE TABLE `user_wsis` (
  `id` bigint(20) UNSIGNED NOT NULL,
  `user_id` bigint(20) UNSIGNED DEFAULT NULL,
  `name` text NOT NULL,
  `store_name` text NOT NULL,
  `type` enum('SVS','DZI') NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Indexes for dumped tables
--

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `username` (`username`);

--
-- Indexes for table `user_augmentations`
--
ALTER TABLE `user_augmentations`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `store_name` (`store_name`) USING HASH,
  ADD KEY `user_id` (`user_id`);

--
-- Indexes for table `user_classifications`
--
ALTER TABLE `user_classifications`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `store_name` (`store_name`) USING HASH,
  ADD KEY `user_id` (`user_id`);

--
-- Indexes for table `user_datasets`
--
ALTER TABLE `user_datasets`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `store_name` (`store_name`) USING HASH,
  ADD KEY `user_id` (`user_id`);

--
-- Indexes for table `user_segmentations`
--
ALTER TABLE `user_segmentations`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `store_name` (`store_name`) USING HASH,
  ADD KEY `user_id` (`user_id`);

--
-- Indexes for table `user_wsis`
--
ALTER TABLE `user_wsis`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY ` store_name` (`store_name`) USING HASH,
  ADD KEY `user_id` (`user_id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `users`
  MODIFY `id` bigint(20) UNSIGNED NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `user_augmentations`
--
ALTER TABLE `user_augmentations`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `user_classifications`
--
ALTER TABLE `user_classifications`
  MODIFY `id` bigint(20) UNSIGNED NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `user_datasets`
--
ALTER TABLE `user_datasets`
  MODIFY `id` bigint(20) UNSIGNED NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `user_segmentations`
--
ALTER TABLE `user_segmentations`
  MODIFY `id` bigint(20) UNSIGNED NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `user_wsis`
--
ALTER TABLE `user_wsis`
  MODIFY `id` bigint(20) UNSIGNED NOT NULL AUTO_INCREMENT;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `user_augmentations`
--
ALTER TABLE `user_augmentations`
  ADD CONSTRAINT `user_augmentations_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE SET NULL ON UPDATE CASCADE;

--
-- Constraints for table `user_classifications`
--
ALTER TABLE `user_classifications`
  ADD CONSTRAINT `user_classifications_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE SET NULL ON UPDATE CASCADE;

--
-- Constraints for table `user_datasets`
--
ALTER TABLE `user_datasets`
  ADD CONSTRAINT `user_datasets_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE SET NULL ON UPDATE CASCADE;

--
-- Constraints for table `user_segmentations`
--
ALTER TABLE `user_segmentations`
  ADD CONSTRAINT `user_segmentations_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE SET NULL ON UPDATE CASCADE;

--
-- Constraints for table `user_wsis`
--
ALTER TABLE `user_wsis`
  ADD CONSTRAINT `user_wsis_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE SET NULL ON UPDATE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
