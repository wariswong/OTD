-- Create admins table
CREATE TABLE IF NOT EXISTS admins (
    id INT AUTO_INCREMENT PRIMARY KEY,
    hr_employee_id VARCHAR(20) NOT NULL UNIQUE,
    role ENUM('admin_system', 'admin_region') NOT NULL DEFAULT 'admin_region',
    region VARCHAR(10) NULL, -- Null for system admins, specified for region admins
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Insert initial system admin
INSERT IGNORE INTO admins (hr_employee_id, role, region) 
VALUES ('505975', 'admin_system', NULL);
